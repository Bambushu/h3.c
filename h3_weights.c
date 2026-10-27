#include "h3_weights.h"

#include <dirent.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
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

/* Find name and check its rank and shape; the caller checks the dtype. */
static const h3_st_tensor *find_shaped(const h3_weight_store *store,
                                       const char *name, int ndim,
                                       const uint64_t *shape,
                                       const h3_st_header **header,
                                       size_t *elements_out,
                                       char *error, size_t error_size) {
    const h3_st_tensor *tensor = h3_weight_find(store, name, header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has rank %d, expected %d", name,
             tensor->ndim, ndim);
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
    *elements_out = (size_t)elements;
    return tensor;
}

static int read_exact(const char *path, uint64_t offset, void *buffer,
                      size_t bytes) {
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) return 0;
    size_t done = 0;
    while (done < bytes) {
        ssize_t count = pread(descriptor, (char *)buffer + done, bytes - done,
                              (off_t)(offset + done));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) break;
        done += (size_t)count;
    }
    close(descriptor);
    return done == bytes;
}

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int h3_weight_int8_scales(const h3_weight_store *store, const char *name,
                          uint64_t rows, const h3_st_header **header,
                          const h3_st_tensor **tensor,
                          char *error, size_t error_size) {
    char scale_name[256];
    snprintf(scale_name, sizeof(scale_name), "%s%s", name,
             H3_INT8_SCALE_SUFFIX);
    uint64_t shape[] = {rows};
    size_t elements = 0;
    *tensor = find_shaped(store, scale_name, 1, shape, header, &elements,
                          error, error_size);
    if (!*tensor) return 0;
    if ((*tensor)->dtype != H3_DTYPE_F32) {
        fail(error, error_size, "int8 scales %s are not F32", scale_name);
        return 0;
    }
    return 1;
}

/* Expand an I8 weight with per-row F32 scales into BF16, the same arithmetic
 * as h3_dequantize_int8_rows_bf16. */
static int expand_int8(const h3_weight_store *store, const char *name,
                       const h3_st_header *header, const h3_st_tensor *tensor,
                       uint16_t *destination, char *error, size_t error_size) {
    uint64_t rows = tensor->shape[0], columns = tensor->shape[1];
    const h3_st_header *scale_header = NULL;
    const h3_st_tensor *scale_tensor = NULL;
    if (!h3_weight_int8_scales(store, name, rows, &scale_header,
                               &scale_tensor, error, error_size)) return 0;
    int8_t *weights = malloc((size_t)(rows * columns));
    float *scales = malloc((size_t)rows * sizeof(float));
    int ok = weights && scales &&
             read_exact(header->path, tensor->file_offset, weights,
                        (size_t)(rows * columns)) &&
             read_exact(scale_header->path, scale_tensor->file_offset, scales,
                        (size_t)rows * sizeof(float));
    if (ok) {
        dispatch_apply((size_t)rows,
                       dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                       ^(size_t row) {
            const int8_t *source = weights + row * columns;
            uint16_t *target = destination + row * columns;
            float scale = scales[row];
            for (uint64_t column = 0; column < columns; column++)
                target[column] = f32_to_bf16((float)source[column] * scale);
        });
    } else {
        fail(error, error_size, "cannot read int8 weight %s", name);
    }
    free(weights);
    free(scales);
    return ok;
}

int h3_weight_read_bf16(const h3_weight_store *store, const char *name,
                        int ndim, const uint64_t *shape,
                        h3_gpu_tensor *destination,
                        char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    size_t elements = 0;
    const h3_st_tensor *tensor = find_shaped(store, name, ndim, shape, &header,
                                             &elements, error, error_size);
    if (!tensor) return 0;
    if (tensor->dtype == H3_DTYPE_I8 && ndim == 2 &&
        h3_gpu_tensor_dtype(destination) == H3_GPU_BF16 &&
        h3_gpu_tensor_elements(destination) == elements)
        return expand_int8(store, name, header, tensor,
                           (uint16_t *)h3_gpu_tensor_contents(destination),
                           error, error_size);
    if (tensor->dtype != H3_DTYPE_BF16) {
        fail(error, error_size, "weight %s has dtype %s, expected BF16 or I8",
             name, h3_dtype_name(tensor->dtype));
        return 0;
    }
    return h3_gpu_tensor_read_file_bf16(destination, header->path,
                                        tensor->file_offset, elements,
                                        error, error_size);
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    size_t elements = 0;
    const h3_st_tensor *tensor = find_shaped(store, name, ndim, shape, &header,
                                             &elements, error, error_size);
    if (!tensor) return NULL;
    if (dtype == H3_DTYPE_BF16 && tensor->dtype == H3_DTYPE_I8 && ndim == 2) {
        h3_gpu_tensor *result = h3_gpu_tensor_new_bf16(gpu, elements);
        if (!result) {
            fail(error, error_size, "cannot allocate %s: %s", name,
                 h3_gpu_error(gpu));
            return NULL;
        }
        if (!expand_int8(store, name, header, tensor,
                         (uint16_t *)h3_gpu_tensor_contents(result),
                         error, error_size)) {
            h3_gpu_tensor_free(result);
            return NULL;
        }
        return result;
    }
    if (tensor->dtype != dtype) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    h3_gpu_tensor *result = dtype == H3_DTYPE_BF16 ?
        h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                elements) :
        h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                               elements);
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
