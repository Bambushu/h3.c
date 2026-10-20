#include "h3_dit_cache.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>

#define HASH_SEED 1469598103934665603ULL

static const char CACHE_MAGIC[8] = {'H','3','D','I','T','C','H','1'};
static const size_t IO_CHUNK = (size_t)1 << 30;

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

uint64_t h3_dit_cache_hash(uint64_t hash, const void *data, size_t size) {
    const uint8_t *bytes = data;
    for (size_t at = 0; at < size; at++) {
        hash ^= bytes[at];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t hash_string(uint64_t hash, const char *text) {
    /* Include the terminator so adjacent strings cannot run together. */
    return h3_dit_cache_hash(hash, text ? text : "", text ? strlen(text) + 1 : 1);
}

static uint64_t hash_u64(uint64_t hash, uint64_t value) {
    return h3_dit_cache_hash(hash, &value, sizeof(value));
}

static int is_safetensors(const struct dirent *entry) {
    size_t length = strlen(entry->d_name);
    return length > 12 && !strcmp(entry->d_name + length - 12,
                                  ".safetensors");
}

uint64_t h3_dit_cache_checkpoint_fingerprint(const char *directory) {
    char *resolved = realpath(directory, NULL);
    const char *root = resolved ? resolved : directory;
    uint64_t hash = hash_string(HASH_SEED, root);
    struct dirent **entries = NULL;
    int count = scandir(root, &entries, is_safetensors, alphasort);
    for (int index = 0; index < count; index++) {
        char path[2048];
        struct stat info;
        snprintf(path, sizeof(path), "%s/%s", root, entries[index]->d_name);
        if (!stat(path, &info)) {
            int64_t fields[] = {(int64_t)info.st_size,
                                (int64_t)info.st_mtimespec.tv_sec,
                                (int64_t)info.st_mtimespec.tv_nsec};
            hash = hash_string(hash, entries[index]->d_name);
            hash = h3_dit_cache_hash(hash, fields, sizeof(fields));
        }
        free(entries[index]);
    }
    free(entries);
    hash = hash_u64(hash, (uint64_t)(int64_t)count);
    free(resolved);
    return hash;
}

static uint64_t hash_file(uint64_t hash, const char *path) {
    FILE *file = path ? fopen(path, "rb") : NULL;
    if (!file) return hash_string(hash, path);
    unsigned char buffer[65536];
    size_t count;
    while ((count = fread(buffer, 1, sizeof(buffer), file)) > 0)
        hash = h3_dit_cache_hash(hash, buffer, count);
    fclose(file);
    return hash;
}

int h3_dit_cache_init(h3_dit_cache *cache, const char *directory,
                      const char *weight_directory,
                      const char *gpu_description,
                      const char *shader_source_path) {
    if (!cache || !directory || !*directory || !weight_directory) return 0;
    memset(cache, 0, sizeof(*cache));
    int length = snprintf(cache->directory, sizeof(cache->directory), "%s",
                          directory);
    if (length <= 0 || (size_t)length >= sizeof(cache->directory)) return 0;
    if (mkdir(directory, 0700) != 0 && errno != EEXIST) return 0;
    struct stat info;
    if (stat(directory, &info) != 0 || !S_ISDIR(info.st_mode)) return 0;

    char *resolved = realpath(weight_directory, NULL);
    cache->source = hash_string(HASH_SEED,
                                resolved ? resolved : weight_directory);
    free(resolved);
    cache->checkpoint = h3_dit_cache_checkpoint_fingerprint(weight_directory);

    /* Anything that can change the bytes a fresh load produces: the Metal
     * device and TensorOps mode, the OS build (Metal compiler, MPSGraph and
     * libm), the shader source and the quantizer kernel selection. */
    char build[256] = {0};
    size_t build_size = sizeof(build) - 1;
    if (sysctlbyname("kern.osversion", build, &build_size, NULL, 0) != 0)
        build[0] = '\0';
    uint64_t variant = hash_u64(HASH_SEED, H3_DIT_CACHE_VERSION);
    variant = hash_string(variant, gpu_description);
    variant = hash_string(variant, build);
    variant = hash_file(variant, shader_source_path);
    variant = hash_u64(variant, getenv("H3_INT8_VECTOR_QUANT") != NULL);
    cache->variant = variant;
    return 1;
}

static uint64_t base_key(const h3_dit_cache *cache, h3_dit_cache_kind kind) {
    uint64_t key = hash_u64(HASH_SEED, H3_DIT_CACHE_VERSION);
    key = hash_u64(key, (uint64_t)kind);
    key = hash_u64(key, cache->checkpoint);
    return hash_u64(key, cache->variant);
}

uint64_t h3_dit_cache_int8_key(const h3_dit_cache *cache, unsigned block,
                               unsigned matrices) {
    uint64_t key = base_key(cache, H3_DIT_CACHE_INT8);
    key = hash_u64(key, block);
    return hash_u64(key, matrices);
}

uint64_t h3_dit_cache_adaln_key(const h3_dit_cache *cache,
                                const h3_sigma_schedule *sigmas,
                                int visual_condition, int audio_condition) {
    uint64_t key = base_key(cache, H3_DIT_CACHE_ADALN);
    int steps = sigmas->steps;
    if (steps < 0) steps = 0;
    if (steps > H3_MAX_STEPS) steps = H3_MAX_STEPS;
    key = hash_u64(key, (uint64_t)steps);
    key = h3_dit_cache_hash(key, sigmas->video,
                            (size_t)steps * sizeof(*sigmas->video));
    key = h3_dit_cache_hash(key, sigmas->audio,
                            (size_t)steps * sizeof(*sigmas->audio));
    key = hash_u64(key, visual_condition != 0);
    return hash_u64(key, audio_condition != 0);
}

static int format_path(char *path, size_t size, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    int length = vsnprintf(path, size, format, arguments);
    va_end(arguments);
    return length > 0 && (size_t)length < size;
}

int h3_dit_cache_int8_path(const h3_dit_cache *cache, unsigned block,
                           uint64_t key, char *path, size_t size) {
    return format_path(path, size, "%s/dit-%016llx-int8-b%02u-%016llx.cache",
                       cache->directory, (unsigned long long)cache->source,
                       block, (unsigned long long)key);
}

int h3_dit_cache_adaln_path(const h3_dit_cache *cache, int steps,
                            uint64_t key, char *path, size_t size) {
    return format_path(path, size, "%s/dit-%016llx-adaln-s%d-%016llx.cache",
                       cache->directory, (unsigned long long)cache->source,
                       steps, (unsigned long long)key);
}

static int align_up(uint64_t value, uint64_t *aligned) {
    uint64_t mask = H3_DIT_CACHE_ALIGN - 1;
    if (value > UINT64_MAX - mask) return 0;
    *aligned = (value + mask) & ~mask;
    return 1;
}

int h3_dit_cache_header_init(h3_dit_cache_header *header,
                             h3_dit_cache_kind kind, uint64_t key,
                             const uint64_t *entry_bytes, unsigned count) {
    if (!header || !entry_bytes || !count ||
        count > H3_DIT_CACHE_MAX_ENTRIES) return 0;
    memset(header, 0, sizeof(*header));
    memcpy(header->magic, CACHE_MAGIC, sizeof(header->magic));
    header->version = H3_DIT_CACHE_VERSION;
    header->kind = (uint32_t)kind;
    header->key = key;
    header->entry_count = count;
    uint64_t cursor;
    if (!align_up(sizeof(*header), &cursor)) return 0;
    for (unsigned index = 0; index < count; index++) {
        if (!entry_bytes[index] || entry_bytes[index] > UINT64_MAX - cursor)
            return 0;
        header->entries[index].offset = cursor;
        header->entries[index].bytes = entry_bytes[index];
        if (!align_up(cursor + entry_bytes[index], &cursor)) return 0;
    }
    if (cursor > (uint64_t)INT64_MAX) return 0;
    header->file_bytes = cursor;
    return 1;
}

static int read_all_at(int descriptor, void *data, size_t size,
                       uint64_t offset) {
    unsigned char *bytes = data;
    while (size) {
        size_t request = size < IO_CHUNK ? size : IO_CHUNK;
        ssize_t count = pread(descriptor, bytes, request, (off_t)offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        bytes += count;
        size -= (size_t)count;
        offset += (uint64_t)count;
    }
    return 1;
}

static int write_all_at(int descriptor, const void *data, uint64_t size,
                        uint64_t offset) {
    const unsigned char *bytes = data;
    while (size) {
        size_t request = size < IO_CHUNK ? (size_t)size : IO_CHUNK;
        ssize_t count = pwrite(descriptor, bytes, request, (off_t)offset);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        bytes += count;
        size -= (uint64_t)count;
        offset += (uint64_t)count;
    }
    return 1;
}

int h3_dit_cache_valid(const char *path, const h3_dit_cache_header *expected) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return 0;
    h3_dit_cache_header header;
    struct stat info;
    int ok = read_all_at(descriptor, &header, sizeof(header), 0) &&
             !memcmp(&header, expected, sizeof(header)) &&
             !fstat(descriptor, &info) &&
             (uint64_t)info.st_size == expected->file_bytes;
    close(descriptor);
    return ok;
}

int h3_dit_cache_write(const char *path, const h3_dit_cache_header *header,
                       const void *const *payloads,
                       char *error, size_t error_size) {
    char temp[1200];
    if (!format_path(temp, sizeof(temp), "%s.tmp.%d", path, (int)getpid())) {
        fail(error, error_size, "cache path is too long: %s", path);
        return 0;
    }
    int descriptor = open(temp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0 && errno == EEXIST) {
        /* Left by an earlier process that had this pid. */
        unlink(temp);
        descriptor = open(temp, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    }
    if (descriptor < 0) {
        fail(error, error_size, "cannot create %s: %s", temp, strerror(errno));
        return 0;
    }
    int ok = fchmod(descriptor, 0600) == 0 &&
             write_all_at(descriptor, header, sizeof(*header), 0);
    for (unsigned index = 0; ok && index < header->entry_count; index++)
        ok = write_all_at(descriptor, payloads[index],
                          header->entries[index].bytes,
                          header->entries[index].offset);
    ok = ok && ftruncate(descriptor, (off_t)header->file_bytes) == 0 &&
         fsync(descriptor) == 0;
    int detail = ok ? 0 : errno;
    if (close(descriptor) != 0 && ok) {
        ok = 0;
        detail = errno;
    }
    if (ok && rename(temp, path) != 0) {
        ok = 0;
        detail = errno;
    }
    if (!ok) {
        unlink(temp);
        fail(error, error_size, "cannot write %s: %s", path,
             detail ? strerror(detail) : "short write");
    }
    return ok;
}

void h3_dit_cache_prune_int8(const h3_dit_cache *cache, unsigned block,
                             const char *keep_path) {
    char prefix[96];
    if (!format_path(prefix, sizeof(prefix), "dit-%016llx-int8-b%02u-",
                     (unsigned long long)cache->source, block)) return;
    size_t prefix_length = strlen(prefix);
    DIR *directory = opendir(cache->directory);
    if (!directory) return;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (strncmp(entry->d_name, prefix, prefix_length)) continue;
        char path[1200];
        if (!format_path(path, sizeof(path), "%s/%s", cache->directory,
                         entry->d_name) ||
            (keep_path && !strcmp(path, keep_path))) continue;
        const char *temp = strstr(entry->d_name, ".cache.tmp.");
        if (temp) {
            char *end = NULL;
            long pid = strtol(temp + strlen(".cache.tmp."), &end, 10);
            if (end && !*end && pid > 0 && pid <= INT_MAX &&
                kill((pid_t)pid, 0) != 0 && errno == ESRCH) unlink(path);
            continue;
        }
        size_t length = strlen(entry->d_name);
        if (length > 6 && !strcmp(entry->d_name + length - 6, ".cache"))
            unlink(path);
    }
    closedir(directory);
}
