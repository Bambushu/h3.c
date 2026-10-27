/* Write an int8 copy of a MiniMax-H3 snapshot: the same folder layout, with
 * the large 2D BF16 weights stored as I8 plus one F32 scale per row in
 * "<name>_int8_scale" (h3_weights.h). h3 expands them to BF16 as it reads:
 * on the CPU for loads and the text encoder, on the GPU for --ssd-streaming.
 *
 * Quantized: DiT block QKV, attention output, FC1, FC2 and AdaLN projections;
 * Qwen language-model layer projections. Dropped: lm_head (unused by h3).
 * Everything else is copied as is.
 *
 * Usage: h3_int8_pack SOURCE_MODEL_DIR DESTINATION_DIR */
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <dirent.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static double started;

static double now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int read_all(int descriptor, void *buffer, size_t bytes, uint64_t at) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = pread(descriptor, (char *)buffer + done, bytes - done,
                              (off_t)(at + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        done += (size_t)count;
    }
    return 1;
}

static int write_all(int descriptor, const void *buffer, size_t bytes) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = write(descriptor, (const char *)buffer + done,
                              bytes - done);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) return 0;
        done += (size_t)count;
    }
    return 1;
}

static int ends_with(const char *text, const char *suffix) {
    size_t length = strlen(text), tail = strlen(suffix);
    return length >= tail && !strcmp(text + length - tail, suffix);
}

static int dropped(const char *name) {
    return !strcmp(name, "lm_head.weight");
}

static int quantized(const h3_st_tensor *tensor) {
    if (tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2) return 0;
    const char *name = tensor->name;
    if (!strncmp(name, "blocks.", 7))
        return ends_with(name, ".attn.qkv_proj.weight") ||
               ends_with(name, ".attn.out_proj.weight") ||
               ends_with(name, ".mlp.fc1.weight") ||
               ends_with(name, ".mlp.fc2.weight") ||
               ends_with(name, ".adaln_proj.linear.weight");
    if (!strncmp(name, "model.language_model.layers.", 28))
        return ends_with(name, "_proj.weight");
    return 0;
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

/* Same arithmetic as h3_quantize_bf16_int8_rows_scalar with clip 1. */
static void quantize_rows(const uint16_t *input, int8_t *output, float *scales,
                          uint64_t rows, uint64_t columns) {
    dispatch_apply((size_t)rows,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t row) {
        const uint16_t *source = input + row * columns;
        int8_t *target = output + row * columns;
        float max_abs = 0.0f;
        for (uint64_t column = 0; column < columns; column++)
            max_abs = fmaxf(max_abs, fabsf(bf16_to_f32(source[column])));
        float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        float inverse = max_abs > 0.0f ? 127.0f / max_abs : 127.0f;
        scales[row] = scale;
        for (uint64_t column = 0; column < columns; column++) {
            int value = (int)rintf(bf16_to_f32(source[column]) * inverse);
            if (value > 127) value = 127;
            if (value < -127) value = -127;
            target[column] = (int8_t)value;
        }
    });
}

typedef struct {
    char *json;
    size_t length;
    size_t capacity;
} text;

static void append(text *out, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

static void append(text *out, const char *format, ...) {
    for (;;) {
        va_list arguments;
        va_start(arguments, format);
        int written = vsnprintf(out->json + out->length,
                                out->capacity - out->length, format, arguments);
        va_end(arguments);
        if (written >= 0 && (size_t)written < out->capacity - out->length) {
            out->length += (size_t)written;
            return;
        }
        out->capacity = out->capacity ? out->capacity * 2 : 1 << 20;
        out->json = realloc(out->json, out->capacity);
        if (!out->json) {
            fprintf(stderr, "h3_int8_pack: out of memory\n");
            exit(1);
        }
    }
}

static int compare_offsets(const void *left, const void *right) {
    const h3_st_tensor *const *a = left, *const *b = right;
    return (*a)->data_begin < (*b)->data_begin ? -1 :
           (*a)->data_begin > (*b)->data_begin;
}

static int pack_shard(const char *source, const char *destination) {
    h3_st_header header;
    char error[512];
    if (!h3_st_read_header(source, &header, error, sizeof(error))) {
        fprintf(stderr, "h3_int8_pack: %s\n", error);
        return 0;
    }
    const h3_st_tensor **order = calloc(header.tensor_count, sizeof(*order));
    size_t count = 0;
    for (size_t index = 0; index < header.tensor_count; index++)
        if (!dropped(header.tensors[index].name))
            order[count++] = &header.tensors[index];
    qsort(order, count, sizeof(*order), compare_offsets);

    text json = {0};
    uint64_t at = 0;
    append(&json, "{\"__metadata__\":{\"format\":\"pt\",\"h3\":\"int8\"}");
    for (size_t index = 0; index < count; index++) {
        const h3_st_tensor *tensor = order[index];
        if (quantized(tensor)) {
            uint64_t rows = tensor->shape[0], columns = tensor->shape[1];
            append(&json, ",\"%s\":{\"dtype\":\"I8\",\"shape\":[%llu,%llu],"
                   "\"data_offsets\":[%llu,%llu]}", tensor->name,
                   (unsigned long long)rows, (unsigned long long)columns,
                   (unsigned long long)at,
                   (unsigned long long)(at + rows * columns));
            at += rows * columns;
            append(&json, ",\"%s%s\":{\"dtype\":\"F32\",\"shape\":[%llu],"
                   "\"data_offsets\":[%llu,%llu]}", tensor->name,
                   H3_INT8_SCALE_SUFFIX, (unsigned long long)rows,
                   (unsigned long long)at,
                   (unsigned long long)(at + rows * sizeof(float)));
            at += rows * sizeof(float);
            continue;
        }
        uint64_t bytes = tensor->data_end - tensor->data_begin;
        append(&json, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[", tensor->name,
               h3_dtype_name(tensor->dtype));
        for (int dimension = 0; dimension < tensor->ndim; dimension++)
            append(&json, "%s%llu", dimension ? "," : "",
                   (unsigned long long)tensor->shape[dimension]);
        append(&json, "],\"data_offsets\":[%llu,%llu]}",
               (unsigned long long)at, (unsigned long long)(at + bytes));
        at += bytes;
    }
    append(&json, "}");
    while (json.length % 8) append(&json, " ");

    char partial[2100];
    snprintf(partial, sizeof(partial), "%s.partial", destination);
    int input = open(source, O_RDONLY | O_CLOEXEC);
    int output = open(partial, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    uint64_t header_bytes = json.length;
    int ok = input >= 0 && output >= 0 &&
             write_all(output, &header_bytes, sizeof(header_bytes)) &&
             write_all(output, json.json, json.length);
    size_t chunk = 64u << 20;
    unsigned char *buffer = malloc(chunk);
    for (size_t index = 0; ok && index < count; index++) {
        const h3_st_tensor *tensor = order[index];
        uint64_t bytes = tensor->data_end - tensor->data_begin;
        if (quantized(tensor)) {
            uint64_t rows = tensor->shape[0], columns = tensor->shape[1];
            uint16_t *values = malloc((size_t)bytes);
            int8_t *weights = malloc((size_t)(rows * columns));
            float *scales = malloc((size_t)rows * sizeof(float));
            ok = values && weights && scales &&
                 read_all(input, values, (size_t)bytes, tensor->file_offset);
            if (ok) {
                quantize_rows(values, weights, scales, rows, columns);
                ok = write_all(output, weights, (size_t)(rows * columns)) &&
                     write_all(output, scales, (size_t)rows * sizeof(float));
            }
            free(values);
            free(weights);
            free(scales);
            continue;
        }
        for (uint64_t done = 0; ok && done < bytes;) {
            size_t step = bytes - done < chunk ? (size_t)(bytes - done) : chunk;
            ok = read_all(input, buffer, step, tensor->file_offset + done) &&
                 write_all(output, buffer, step);
            done += step;
        }
    }
    free(buffer);
    if (input >= 0) close(input);
    if (output >= 0 && close(output) != 0) ok = 0;
    if (ok && rename(partial, destination) != 0) ok = 0;
    if (!ok) fprintf(stderr, "h3_int8_pack: cannot write %s: %s\n",
                     destination, strerror(errno));
    free(json.json);
    free(order);
    h3_st_free_header(&header);
    return ok;
}

static int copy_file(const char *source, const char *destination) {
    int input = open(source, O_RDONLY | O_CLOEXEC);
    int output = open(destination, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                      0644);
    unsigned char buffer[1 << 16];
    ssize_t count = 0;
    int ok = input >= 0 && output >= 0;
    while (ok && (count = read(input, buffer, sizeof(buffer))) > 0)
        ok = write_all(output, buffer, (size_t)count);
    if (count < 0) ok = 0;
    if (input >= 0) close(input);
    if (output >= 0 && close(output) != 0) ok = 0;
    if (!ok) fprintf(stderr, "h3_int8_pack: cannot copy %s\n", source);
    return ok;
}

/* Mirror a directory, following the snapshot's blob symlinks. */
static int pack_tree(const char *source, const char *destination) {
    if (mkdir(destination, 0755) != 0 && errno != EEXIST) {
        fprintf(stderr, "h3_int8_pack: cannot create %s: %s\n", destination,
                strerror(errno));
        return 0;
    }
    DIR *directory = opendir(source);
    if (!directory) {
        fprintf(stderr, "h3_int8_pack: cannot open %s\n", source);
        return 0;
    }
    struct dirent *entry;
    int ok = 1;
    while (ok && (entry = readdir(directory)) != NULL) {
        if (entry->d_name[0] == '.') continue;
        char from[2048], to[2048];
        snprintf(from, sizeof(from), "%s/%s", source, entry->d_name);
        snprintf(to, sizeof(to), "%s/%s", destination, entry->d_name);
        struct stat info;
        if (stat(from, &info) != 0) continue;
        if (S_ISDIR(info.st_mode)) {
            ok = pack_tree(from, to);
        } else if (ends_with(entry->d_name, ".safetensors")) {
            ok = pack_shard(from, to);
            fprintf(stderr, "%7.0fs  %s\n", now() - started, to);
        } else if (!ends_with(entry->d_name, ".safetensors.index.json")) {
            ok = copy_file(from, to);
        }
    }
    closedir(directory);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s SOURCE_MODEL_DIR DESTINATION_DIR\n",
                argv[0]);
        return 2;
    }
    started = now();
    return pack_tree(argv[1], argv[2]) ? 0 : 1;
}
