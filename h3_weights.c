#define ACCELERATE_NEW_LAPACK
#define ACCELERATE_LAPACK_ILP64
#include "h3_weights.h"

#include <Accelerate/Accelerate.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    char *target;
    const h3_st_tensor *a;  /* [rank, columns] */
    const h3_st_tensor *b;  /* [rows, rank] */
} h3_lora_pair;

typedef struct {
    h3_st_header header;
    h3_lora_pair *pairs;
    size_t count;
    float scale;
    uint64_t key;
} h3_lora_adapter;

/* A stack of adapters; one adapter keeps its own key, so a single-LoRA int8
 * cache stays valid. */
struct h3_lora {
    h3_lora_adapter *adapters;
    size_t count;
    size_t pairs;
    uint64_t key;
};

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
    const h3_lora *lora;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int compare_pairs(const void *left, const void *right) {
    const h3_lora_pair *a = left;
    const h3_lora_pair *b = right;
    return strcmp(a->target, b->target);
}

static const h3_lora_pair *adapter_pair(const h3_lora_adapter *adapter,
                                        const char *name) {
    h3_lora_pair probe = {(char *)name, NULL, NULL};
    return bsearch(&probe, adapter->pairs, adapter->count,
                   sizeof(*adapter->pairs), compare_pairs);
}

static int find_pair(const h3_lora *lora, const char *name) {
    if (!lora || !name) return 0;
    for (size_t index = 0; index < lora->count; index++)
        if (adapter_pair(&lora->adapters[index], name)) return 1;
    return 0;
}

static int lora_factor(const h3_st_tensor *tensor) {
    return tensor->ndim == 2 && tensor->shape[0] && tensor->shape[1] &&
           (tensor->dtype == H3_DTYPE_BF16 || tensor->dtype == H3_DTYPE_F32);
}

/* FNV-1a over 64-bit words of the file, then its length and the scale. A
 * cache key, not a checksum: it only has to change when the adapter does. */
static int hash_file_words(const char *path, uint64_t *hash,
                           char *error, size_t error_size) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    struct stat info;
    if (descriptor < 0 || fstat(descriptor, &info) != 0 || info.st_size <= 0) {
        if (descriptor >= 0) close(descriptor);
        fail(error, error_size, "cannot open LoRA %s", path);
        return 0;
    }
    size_t length = (size_t)info.st_size;
    const unsigned char *bytes = mmap(NULL, length, PROT_READ, MAP_PRIVATE,
                                      descriptor, 0);
    close(descriptor);
    if (bytes == MAP_FAILED) {
        fail(error, error_size, "cannot map LoRA %s", path);
        return 0;
    }
    uint64_t value = UINT64_C(14695981039346656037);
    size_t at = 0;
    for (; at + sizeof(uint64_t) <= length; at += sizeof(uint64_t)) {
        uint64_t word;
        memcpy(&word, bytes + at, sizeof(word));
        value = (value ^ word) * UINT64_C(1099511628211);
    }
    for (; at < length; at++)
        value = (value ^ bytes[at]) * UINT64_C(1099511628211);
    munmap((void *)bytes, length);
    *hash = (value ^ (uint64_t)length) * UINT64_C(1099511628211);
    return 1;
}

static void adapter_free(h3_lora_adapter *adapter) {
    for (size_t index = 0; index < adapter->count; index++)
        free(adapter->pairs[index].target);
    free(adapter->pairs);
    h3_st_free_header(&adapter->header);
}

/* ComfyUI-format adapters name targets diffusion_model.<key>; the checkpoint
 * keys have no prefix. */
static const char *strip_prefix(const char *name) {
    static const char prefix[] = "diffusion_model.";
    return strncmp(name, prefix, sizeof(prefix) - 1) ? name
                                                     : name + sizeof(prefix) - 1;
}

static int adapter_open(h3_lora_adapter *lora, const char *path, float scale,
                        char *error, size_t error_size) {
    memset(lora, 0, sizeof(*lora));
    if (!path || !*path || !isfinite(scale) || scale == 0.0f) {
        fail(error, error_size, "LoRA needs a path and a finite nonzero scale");
        return 0;
    }
    lora->scale = scale;
    char detail[384];
    if (!h3_st_read_header(path, &lora->header, detail, sizeof(detail))) {
        fail(error, error_size, "%s", detail);
        return 0;
    }
    static const char suffix_a[] = ".lora_A.weight";
    static const char suffix_b[] = ".lora_B.weight";
    size_t length_a = sizeof(suffix_a) - 1;
    lora->pairs = calloc(lora->header.tensor_count ? lora->header.tensor_count : 1,
                         sizeof(*lora->pairs));
    if (!lora->pairs) goto out_of_memory;
    for (size_t index = 0; index < lora->header.tensor_count; index++) {
        const h3_st_tensor *a = &lora->header.tensors[index];
        size_t length = strlen(a->name);
        if (length <= length_a || strcmp(a->name + length - length_a, suffix_a))
            continue;
        size_t stem = length - length_a;
        char *name = malloc(stem + sizeof(suffix_b));
        if (!name) goto out_of_memory;
        memcpy(name, a->name, stem);
        memcpy(name + stem, suffix_b, sizeof(suffix_b));
        const h3_st_tensor *b = h3_st_find(&lora->header, name);
        if (!b || !lora_factor(a) || !lora_factor(b) ||
            b->shape[1] != a->shape[0]) {
            fail(error, error_size, "LoRA %s: %s has no matching %s pair",
                 path, a->name, b ? "rank" : suffix_b + 1);
            free(name);
            adapter_free(lora);
            return 0;
        }
        memcpy(name + stem, ".weight", sizeof(".weight"));
        const char *target = strip_prefix(name);
        if (target != name) memmove(name, target, strlen(target) + 1);
        lora->pairs[lora->count++] = (h3_lora_pair){name, a, b};
    }
    if (!lora->count) {
        fail(error, error_size, "LoRA %s has no .lora_A/.lora_B pairs", path);
        adapter_free(lora);
        return 0;
    }
    qsort(lora->pairs, lora->count, sizeof(*lora->pairs), compare_pairs);
    for (size_t index = 1; index < lora->count; index++) {
        if (!strcmp(lora->pairs[index - 1].target, lora->pairs[index].target)) {
            fail(error, error_size, "LoRA %s targets %s twice", path,
                 lora->pairs[index].target);
            adapter_free(lora);
            return 0;
        }
    }
    uint64_t hash;
    if (!hash_file_words(path, &hash, error, error_size)) {
        adapter_free(lora);
        return 0;
    }
    uint32_t bits;
    memcpy(&bits, &scale, sizeof(bits));
    lora->key = (hash ^ bits) * UINT64_C(1099511628211);
    return 1;

out_of_memory:
    fail(error, error_size, "out of memory opening LoRA");
    adapter_free(lora);
    return 0;
}

h3_lora *h3_lora_open(const char *path, float scale,
                      char *error, size_t error_size) {
    h3_lora *lora = calloc(1, sizeof(*lora));
    if (!lora) {
        fail(error, error_size, "out of memory opening LoRA");
        return NULL;
    }
    if (!h3_lora_add(lora, path, scale, error, error_size)) {
        free(lora);
        return NULL;
    }
    return lora;
}

int h3_lora_add(h3_lora *lora, const char *path, float scale,
                char *error, size_t error_size) {
    if (!lora) return 0;
    h3_lora_adapter adapter;
    if (!adapter_open(&adapter, path, scale, error, error_size)) return 0;
    h3_lora_adapter *grown = realloc(lora->adapters,
                                     (lora->count + 1) * sizeof(*grown));
    if (!grown) {
        fail(error, error_size, "out of memory opening LoRA");
        adapter_free(&adapter);
        return 0;
    }
    lora->adapters = grown;
    lora->adapters[lora->count++] = adapter;
    lora->pairs += adapter.count;
    /* Order matters (the deltas sum in order), so the key does too. */
    lora->key = lora->count == 1 ? adapter.key
        : (lora->key ^ adapter.key) * UINT64_C(1099511628211);
    return 1;
}

void h3_lora_free(h3_lora *lora) {
    if (!lora) return;
    for (size_t index = 0; index < lora->count; index++)
        adapter_free(&lora->adapters[index]);
    free(lora->adapters);
    free(lora);
}

size_t h3_lora_pairs(const h3_lora *lora) {
    return lora ? lora->pairs : 0;
}

uint64_t h3_lora_key(const h3_lora *lora) {
    return lora ? lora->key : 0;
}

static float bf16_value(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

/* Round to nearest even, as fold_turbo_lora.py's f32_to_bf16_bytes. */
static uint16_t bf16_round(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

static float *read_factor(const h3_lora_adapter *lora, const h3_st_tensor *tensor,
                          char *error, size_t error_size) {
    uint64_t elements = h3_st_tensor_elements(tensor);
    size_t item = h3_dtype_size(tensor->dtype);
    if (elements > SIZE_MAX / sizeof(float)) return NULL;
    float *values = malloc((size_t)elements * sizeof(float));
    if (!values) {
        fail(error, error_size, "out of memory reading %s", tensor->name);
        return NULL;
    }
    if (!h3_st_read_data(&lora->header, tensor, values,
                         (size_t)elements * item, error, error_size)) {
        free(values);
        return NULL;
    }
    if (tensor->dtype == H3_DTYPE_BF16) {
        const uint16_t *bits = (const uint16_t *)values;
        for (size_t index = (size_t)elements; index-- > 0;)
            values[index] = bf16_value(bits[index]);
    }
    return values;
}

int h3_lora_apply(const h3_lora *lora, const char *name, uint16_t *values,
                  uint64_t rows, uint64_t columns,
                  char *error, size_t error_size) {
    if (!find_pair(lora, name)) return 0;
    if (!values || rows > SIZE_MAX / columns ||
        rows * columns > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "LoRA target %s has the wrong shape", name);
        return -1;
    }
    size_t elements = (size_t)(rows * columns);
    float *sum = calloc(elements, sizeof(float));
    float *delta = sum ? malloc(elements * sizeof(float)) : NULL;
    if (!delta) {
        fail(error, error_size, "out of memory adapting %s", name);
        free(sum);
        return -1;
    }
    for (size_t at = 0; at < lora->count; at++) {
        const h3_lora_adapter *adapter = &lora->adapters[at];
        const h3_lora_pair *pair = adapter_pair(adapter, name);
        if (!pair) continue;
        uint64_t rank = pair->a->shape[0];
        if (pair->b->shape[0] != rows || pair->a->shape[1] != columns) {
            fail(error, error_size, "LoRA target %s has the wrong shape", name);
            free(sum);
            free(delta);
            return -1;
        }
        float *a = read_factor(adapter, pair->a, error, error_size);
        float *b = a ? read_factor(adapter, pair->b, error, error_size) : NULL;
        if (!b) {
            free(a);
            free(sum);
            free(delta);
            return -1;
        }
        /* One full-matrix SGEMM, as numpy's B @ A, so the F32 delta and hence
         * the rounded BF16 weight match the offline fold bit for bit. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (__LAPACK_int)rows, (__LAPACK_int)columns, (__LAPACK_int)rank,
                    1.0f, b, (__LAPACK_int)rank, a, (__LAPACK_int)columns,
                    0.0f, delta, (__LAPACK_int)columns);
        float scale = adapter->scale;
        for (size_t index = 0; index < elements; index++)
            sum[index] += scale * delta[index];
        free(a);
        free(b);
    }
    /* Every delta sums in F32 and the weight rounds to BF16 once; with one
     * adapter 0 + scale * delta is exact, the single-LoRA result. */
    for (size_t index = 0; index < elements; index++) {
        float adapted = bf16_value(values[index]) + sum[index];
        values[index] = bf16_round(adapted);
    }
    free(sum);
    free(delta);
    return 1;
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

h3_weight_store *h3_weight_store_open_lora(const char *directory,
                                           const h3_lora *lora,
                                           char *error, size_t error_size) {
    h3_weight_store *store = h3_weight_store_open(directory, error, error_size);
    if (!store || !lora) return store;
    for (size_t at = 0; at < lora->count; at++)
    for (size_t index = 0; index < lora->adapters[at].count; index++) {
        const h3_lora_pair *pair = &lora->adapters[at].pairs[index];
        const h3_st_tensor *tensor = h3_weight_find(store, pair->target, NULL);
        if (!tensor || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
            tensor->shape[0] != pair->b->shape[0] ||
            tensor->shape[1] != pair->a->shape[1]) {
            fail(error, error_size, "LoRA target %s %s %s: wrong checkpoint "
                 "or adapter?", pair->target,
                 tensor ? "does not match a BF16 matrix in" : "is absent from",
                 directory);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    store->lora = lora;
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    return NULL;
}

/* An adapter target is read sequentially into its own buffer rather than
 * mapped: rewriting a zero-copy mapping faults every page in from the file
 * one at a time, about 25x slower than one streamed read. */
static h3_gpu_tensor *load_adapted(const h3_weight_store *store, h3_gpu *gpu,
                                   const h3_st_header *header,
                                   const h3_st_tensor *tensor,
                                   const char *name, const uint64_t *shape,
                                   size_t elements,
                                   char *error, size_t error_size) {
    h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, elements);
    if (!result) {
        fail(error, error_size, "cannot allocate %s: %s", name,
             h3_gpu_error(gpu));
        return NULL;
    }
    if (!h3_gpu_tensor_stream_file_bf16(result, header->path,
                                        tensor->file_offset, elements,
                                        error, error_size) ||
        h3_lora_apply(store->lora, name,
                      (uint16_t *)h3_gpu_tensor_contents(result),
                      shape[0], shape[1], error, error_size) != 1) {
        h3_gpu_tensor_free(result);
        return NULL;
    }
    return result;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    if (dtype == H3_DTYPE_BF16 && ndim == 2 && find_pair(store->lora, name))
        return load_adapted(store, gpu, header, tensor, name, shape,
                            (size_t)elements, error, error_size);
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                (size_t)elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               (size_t)elements);
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}
