#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
/* int8 checkpoints (tools/h3_int8_pack): a 2D weight may be stored as I8
 * with one F32 scale per row in "<name>_int8_scale". BF16 loads and reads
 * expand it to BF16 (value = q * scale, rounded to nearest even). */
#define H3_INT8_SCALE_SUFFIX "_int8_scale"

int h3_weight_int8_scales(const h3_weight_store *store, const char *name,
                          uint64_t rows, const h3_st_header **header,
                          const h3_st_tensor **tensor,
                          char *error, size_t error_size);

/* Fill an existing shared BF16 tensor with a BF16 or I8 weight. */
int h3_weight_read_bf16(const h3_weight_store *store, const char *name,
                        int ndim, const uint64_t *shape,
                        h3_gpu_tensor *destination,
                        char *error, size_t error_size);

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
