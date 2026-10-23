#ifndef H3_DIT_CACHE_H
#define H3_DIT_CACHE_H

#include "h3_host.h"

#include <stddef.h>
#include <stdint.h>

/* H3_DIT_CACHE=<directory>: persist what a DiT load derives from the
 * checkpoint on the GPU, namely the runtime int8 block weights with their
 * per-row F32 scales and the AdaLN schedule, so a later load maps the exact
 * bytes instead of touching the BF16 sources. Bump the version whenever the
 * quantizer, the AdaLN precompute or this file format changes. */
#define H3_DIT_CACHE_VERSION 1u
#define H3_DIT_CACHE_ALIGN 16384u
#define H3_DIT_CACHE_MAX_ENTRIES 64u

typedef enum {
    H3_DIT_CACHE_INT8 = 1,
    H3_DIT_CACHE_ADALN = 2
} h3_dit_cache_kind;

typedef struct {
    uint64_t offset;
    uint64_t bytes;
} h3_dit_cache_entry;

/* Native-endian header at offset 0. Payloads start at H3_DIT_CACHE_ALIGN
 * boundaries so every entry maps zero-copy. A file is valid only if its header
 * equals the one the reader expects byte for byte and its size matches. */
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t kind;
    uint64_t key;
    uint64_t file_bytes;
    uint32_t entry_count;
    uint32_t reserved;
    h3_dit_cache_entry entries[H3_DIT_CACHE_MAX_ENTRIES];
} h3_dit_cache_header;

typedef struct {
    char directory[1024];
    uint64_t source;      /* resolved checkpoint directory; names the files */
    uint64_t checkpoint;  /* shard names, sizes and modification times */
    uint64_t variant;     /* device, OS build, shader source, quantizer env */
    uint64_t adapter;     /* h3_lora_key of a load-time LoRA, 0 for none */
    int verify;           /* H3_DIT_CACHE_VERIFY */
    int write_failed;     /* stop rewriting after the first failure */
    /* H3_PROFILE counters for one load. */
    unsigned int8_hits;
    unsigned int8_misses;
    unsigned int8_writes;
    int adaln_hit;
    int adaln_written;
    uint64_t read_bytes;
    uint64_t written_bytes;
    double write_seconds;
    double verify_seconds;
} h3_dit_cache;

uint64_t h3_dit_cache_hash(uint64_t hash, const void *data, size_t size);

/* Cheap staleness check, not a content hash: FNV-1a over the resolved
 * directory and each *.safetensors shard's name, size and mtime, the same
 * rule H3_VAE_CACHE uses. */
uint64_t h3_dit_cache_checkpoint_fingerprint(const char *directory);

/* Prepare a cache in directory (created with mode 0700 when absent) for the
 * checkpoint in weight_directory. gpu_description comes from h3_gpu_describe;
 * the shader source is hashed by content. Returns 0 if the directory is not
 * usable, in which case the caller loads without a cache. */
int h3_dit_cache_init(h3_dit_cache *cache, const char *directory,
                      const char *weight_directory,
                      const char *gpu_description,
                      const char *shader_source_path);

/* Key every file on a LoRA applied at load (h3_lora_key). It also renames the
 * files, so base and adapted caches of one checkpoint coexist and pruning one
 * never removes the other. Call right after h3_dit_cache_init. */
void h3_dit_cache_set_adapter(h3_dit_cache *cache, uint64_t adapter);

/* matrices is a bit mask of the int8 matrices stored for this block. Keys are
 * per block, so every active-layer selection shares the same files. */
uint64_t h3_dit_cache_int8_key(const h3_dit_cache *cache, unsigned block,
                               unsigned matrices);
uint64_t h3_dit_cache_adaln_key(const h3_dit_cache *cache,
                                const h3_sigma_schedule *sigmas,
                                int visual_condition, int audio_condition);
int h3_dit_cache_int8_path(const h3_dit_cache *cache, unsigned block,
                           uint64_t key, char *path, size_t size);
int h3_dit_cache_adaln_path(const h3_dit_cache *cache, int steps,
                            uint64_t key, char *path, size_t size);

/* Lay out count payloads of the given sizes. Returns 0 on overflow. */
int h3_dit_cache_header_init(h3_dit_cache_header *header,
                             h3_dit_cache_kind kind, uint64_t key,
                             const uint64_t *entry_bytes, unsigned count);
int h3_dit_cache_valid(const char *path, const h3_dit_cache_header *expected);

/* Write header and payloads to a private (0600) temporary file in the same
 * directory, fsync it and rename it over path. */
int h3_dit_cache_write(const char *path, const h3_dit_cache_header *header,
                       const void *const *payloads,
                       char *error, size_t error_size);

/* Remove superseded int8 files for this checkpoint directory and block (a
 * different key after an OS, shader or checkpoint change) and temporary files
 * left by writers that no longer run. keep_path is never removed. */
void h3_dit_cache_prune_int8(const h3_dit_cache *cache, unsigned block,
                             const char *keep_path);

#endif
