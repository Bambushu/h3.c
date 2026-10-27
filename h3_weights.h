#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* A low-rank adapter applied to BF16 weights as they load, so one checkpoint
 * serves both the base model and an adapted one (the Turbo step-distillation
 * LoRA): W' = W + scale * (B @ A), computed in F32 with one Accelerate SGEMM
 * per tensor and rounded to BF16 nearest-even, the same path as
 * tools/fold_turbo_lora.py. Tensors are named <target>.lora_A.weight [rank,
 * in] and <target>.lora_B.weight [out, rank] over the checkpoint's key space,
 * BF16 or F32. Read-only after open; one adapter may serve several stores. */
typedef struct h3_lora h3_lora;

h3_lora *h3_lora_open(const char *path, float scale,
                      char *error, size_t error_size);
/* Stack another adapter: its scaled delta sums with the others in F32 before
 * the one BF16 round. ComfyUI's diffusion_model. key prefix is stripped. */
int h3_lora_add(h3_lora *lora, const char *path, float scale,
                char *error, size_t error_size);
void h3_lora_free(h3_lora *lora);
size_t h3_lora_pairs(const h3_lora *lora);
/* Hash of the adapter files' bytes and scales in order, for derived-weight
 * caches; a single adapter keeps its own key. */
uint64_t h3_lora_key(const h3_lora *lora);
/* Add the adapter's delta to a row-major BF16 [rows, columns] weight in
 * place. Returns 1 if name is a target, 0 if not, -1 on error. */
int h3_lora_apply(const h3_lora *lora, const char *name, uint16_t *values,
                  uint64_t rows, uint64_t columns,
                  char *error, size_t error_size);

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
/* As above, with every BF16 load of an adapter target adapted. Fails unless
 * each target is a BF16 matrix of this store with the adapter's shape. */
h3_weight_store *h3_weight_store_open_lora(const char *directory,
                                           const h3_lora *lora,
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
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
