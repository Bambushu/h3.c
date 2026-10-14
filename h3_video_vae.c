#include "h3_video_vae.h"

#include "h3_weights.h"

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    LATENT_CHANNELS = 24,
    CHUNK_LATENT_TIME = 7,
    FIRST_CHUNK_FRAMES = 22,
    FRAME_OFFSET = 3,
    HIDDEN = 2048,
    LAYERS = 36,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    REGISTERS = 4,
    SUFFIX = 5,
    ROPE_HALF = 24,
    OUTPUT_PATCH = 3 * 4 * 16 * 16,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

/* H3_VAE_PRECISION selects the arithmetic of the four large block matmuls.
 * The released checkpoint is F32 and stays the default; bf16/int8 keep the
 * residual stream, norms, RoPE and attention in F32 and only lower the
 * QKV/output/MLP projections onto the M5 TensorOps paths. */
enum { VAE_F32 = 0, VAE_BF16 = 1, VAE_INT8 = 2 };
enum { LP_QKV = 0, LP_OUT = 1, LP_W1 = 2, LP_W2 = 3, LP_COUNT = 4 };

/* H3_VAE_CACHE=<directory>: with bf16 precision, the resident decoder reads the
 * lowered projections from this file (about 4.8 GiB) instead of reading
 * 9.7 GiB of F32 and casting. A missing or stale file (weight fingerprint
 * mismatch) is rebuilt from the F32 weights during that load. */
static const size_t LOWP_ELEMENTS[LP_COUNT] = {
    (size_t)INNER * 3 * HIDDEN, (size_t)HIDDEN * INNER,
    (size_t)FFN * 2 * HIDDEN, (size_t)HIDDEN * FFN
};
static const char VAE_CACHE_MAGIC[8] = {'H','3','V','A','E','B','F','1'};

typedef struct {
    char magic[8];
    uint32_t layers;
    uint32_t reserved;
    uint8_t fingerprint[32];
} vae_cache_header;

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *scale1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *w1;
    h3_gpu_tensor *w1_b;
    h3_gpu_tensor *w2;
    h3_gpu_tensor *w2_b;
    h3_gpu_tensor *scale2;
    h3_gpu_tensor *lowp[LP_COUNT];
    h3_gpu_tensor *lowp_scales[LP_COUNT];
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    vae_block blocks[LAYERS];
    h3_gpu_tensor *post_w;
    h3_gpu_tensor *post_b;
    h3_gpu_tensor *embed_w;
    h3_gpu_tensor *embed_b;
    h3_gpu_tensor *registers;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *proj_w;
    h3_gpu_tensor *proj_b;
    h3_gpu_tensor *latent;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *projected;
    h3_gpu_tensor *lowp_in;
    h3_gpu_tensor *lowp_out;
    h3_gpu_tensor *quant_in;
    h3_gpu_tensor *quant_scales;
    h3_gpu_tensor *query_bf16;
    h3_gpu_tensor *key_bf16;
    h3_gpu_tensor *value_bf16;
    int precision;
    int bf16_attention;
    int fused;                /* fused low-precision block epilogues */
    const char *cache_path;   /* read lowered weights from here */
    FILE *cache_out;          /* or append freshly lowered weights here */
    uint16_t *cache_host;
    char cache_file[1024];
    char weight_path[1024];   /* weight directory with links resolved */
    uint32_t patches;
    uint32_t sequence;
    int latent_h;
    int latent_w;
    int latent_t;
    int output_frames;
} vae_context;

struct h3_video_vae_decoder {
    vae_context vae;
    tile_axis y_axis;
    tile_axis x_axis;
    int latent_h;
    int latent_w;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
    /* Pipelined decode: two latent/output slots let tile N+1 run on the GPU
     * while the host unpacks and stitches tile N (the first output slot is
     * vae.projected), and one zero suffix token serves every tile. */
    h3_gpu_tensor *slot_latent[2];
    h3_gpu_tensor *slot_projected;
    h3_gpu_tensor *suffix_zero;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(vae_context *vae, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(vae->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *load_f32(vae_context *vae, const char *name, int ndim,
                               const uint64_t *shape, char *error,
                               size_t error_size) {
    return h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape,
                              error, error_size);
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return load_f32(vae, name, 2, shape, error, error_size);
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
    for (int index = 0; index < LP_COUNT; index++) {
        free_tensor(&block->lowp[index]);
        free_tensor(&block->lowp_scales[index]);
    }
}

static void cleanup(vae_context *vae) {
    if (!vae) return;
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->norm_out_w);
    free_tensor(&vae->norm_out_b); free_tensor(&vae->proj_w);
    free_tensor(&vae->proj_b); free_tensor(&vae->latent); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    free_tensor(&vae->lowp_in); free_tensor(&vae->lowp_out);
    free_tensor(&vae->quant_in); free_tensor(&vae->quant_scales);
    free_tensor(&vae->query_bf16); free_tensor(&vae->key_bf16);
    free_tensor(&vae->value_bf16);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

static int configured_precision(h3_gpu *gpu) {
    const char *value = getenv("H3_VAE_PRECISION");
    if (!value || !*value || !strcmp(value, "f32")) return VAE_F32;
    if (!strcmp(value, "bf16")) return VAE_BF16;
    if (!strcmp(value, "int8"))
        return h3_gpu_has_int8_mlp(gpu) ? VAE_INT8 : VAE_BF16;
    fprintf(stderr, "h3: ignoring unknown H3_VAE_PRECISION=%s\n", value);
    return VAE_F32;
}

/* Replace one resident F32 weight [rows, columns] by its BF16 copy or by
 * int8 rows with one F32 scale per output channel. */
static int lower_weight(vae_context *vae, h3_gpu_tensor **weight,
                        uint32_t rows, uint32_t columns,
                        h3_gpu_tensor **lowp, h3_gpu_tensor **scales,
                        char *error, size_t error_size) {
    size_t count = (size_t)rows * columns;
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(vae->gpu, count);
    int ok = bf16 && h3_gpu_begin(vae->gpu) &&
             h3_gpu_cast_f32_to_bf16(vae->gpu, bf16, *weight,
                                     (uint32_t)count);
    if (ok && vae->precision == VAE_INT8) {
        *lowp = h3_gpu_tensor_new_i8(vae->gpu, count);
        *scales = h3_gpu_tensor_new_f32(vae->gpu, rows);
        ok = *lowp && *scales &&
             h3_gpu_quantize_weight_int8(vae->gpu, *lowp, *scales, bf16,
                                         rows, columns);
    } else if (ok) {
        *lowp = bf16;
        bf16 = NULL;
    }
    ok = ok && h3_gpu_submit(vae->gpu);
    h3_gpu_tensor_free(bf16);
    free_tensor(weight);
    if (!ok)
        fail(error, error_size, "cannot lower video VAE weight: %s",
             h3_gpu_error(vae->gpu));
    return ok;
}

static int lower_block(vae_context *vae, vae_block *block, char *error,
                       size_t error_size) {
    return lower_weight(vae, &block->qkv_w, INNER * 3, HIDDEN,
                        &block->lowp[LP_QKV], &block->lowp_scales[LP_QKV],
                        error, error_size) &&
           lower_weight(vae, &block->out_w, HIDDEN, INNER,
                        &block->lowp[LP_OUT], &block->lowp_scales[LP_OUT],
                        error, error_size) &&
           lower_weight(vae, &block->w1, FFN * 2, HIDDEN,
                        &block->lowp[LP_W1], &block->lowp_scales[LP_W1],
                        error, error_size) &&
           lower_weight(vae, &block->w2, HIDDEN, FFN,
                        &block->lowp[LP_W2], &block->lowp_scales[LP_W2],
                        error, error_size);
}

static size_t cache_block_bytes(void) {
    size_t total = 0;
    for (int which = 0; which < LP_COUNT; which++)
        total += LOWP_ELEMENTS[which] * sizeof(uint16_t);
    return total;
}

static int read_cached_block(vae_context *vae, int index, char *error,
                             size_t error_size) {
    vae_block *block = &vae->blocks[index];
    uint64_t offset = sizeof(vae_cache_header) +
                      (uint64_t)index * cache_block_bytes();
    for (int which = 0; which < LP_COUNT; which++) {
        block->lowp[which] = h3_gpu_tensor_new_bf16(vae->gpu,
                                                    LOWP_ELEMENTS[which]);
        if (!block->lowp[which] ||
            !h3_gpu_tensor_read_file_bf16(block->lowp[which],
                                          vae->cache_path, offset,
                                          LOWP_ELEMENTS[which], error,
                                          error_size)) {
            if (!block->lowp[which])
                fail(error, error_size, "cannot allocate cached video VAE "
                     "weight: %s", h3_gpu_error(vae->gpu));
            return 0;
        }
        offset += LOWP_ELEMENTS[which] * sizeof(uint16_t);
    }
    return 1;
}

/* Best effort: a write failure only drops the cache, never the decode. */
static void append_cached_block(vae_context *vae, const vae_block *block) {
    for (int which = 0; which < LP_COUNT && vae->cache_out; which++) {
        size_t count = LOWP_ELEMENTS[which];
        if (!h3_gpu_tensor_read_bf16(block->lowp[which], vae->cache_host,
                                     count) ||
            fwrite(vae->cache_host, sizeof(uint16_t), count,
                   vae->cache_out) != count) {
            fprintf(stderr, "h3: warning: cannot write video VAE cache\n");
            fclose(vae->cache_out);
            vae->cache_out = NULL;
        }
    }
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F1(qkv_b, "attn.to_qkv.bias", INNER * 3);
    F1(out_b, "attn.to_out.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F1(w1_b, "ff.w1.bias", FFN * 2);
    F1(w2_b, "ff.w2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
    if (vae->cache_path) return read_cached_block(vae, index, error,
                                                  error_size);
    F2(qkv_w, "attn.to_qkv.weight", INNER * 3, HIDDEN);
    F2(out_w, "attn.to_out.weight", HIDDEN, INNER);
    F2(w1, "ff.w1.weight", FFN * 2, HIDDEN);
    F2(w2, "ff.w2.weight", HIDDEN, FFN);
#undef F1
#undef F2
    if (vae->precision == VAE_F32) return 1;
    if (!lower_block(vae, block, error, error_size)) return 0;
    if (vae->cache_out) append_cached_block(vae, block);
    return 1;
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
    vae->post_w = load_f32(vae, "post_quant_conv.weight", 5, post_shape,
                           error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                      LATENT_CHANNELS, error, error_size);
    vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                      error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_f32(vae, "decoder.register_tokens", 3,
                              register_shape, error, error_size);
    return vae->post_w && vae->post_b && vae->embed_w && vae->embed_b &&
           vae->registers;
}

static int load_output_weights(vae_context *vae, char *error,
                               size_t error_size) {
    vae->norm_out_w = f1(vae, "decoder.norm_out.weight", HIDDEN,
                          error, error_size);
    vae->norm_out_b = f1(vae, "decoder.norm_out.bias", HIDDEN,
                          error, error_size);
    vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH, HIDDEN,
                      error, error_size);
    vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                      error, error_size);
    return vae->norm_out_w && vae->norm_out_b && vae->proj_w && vae->proj_b;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory padding video latent");
        return 0;
    }
    size_t row = 0;
    for (int time = 0; time < CHUNK_LATENT_TIME; time++) {
        int source_t = time < vae->latent_t ? time : vae->latent_t - 1;
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)vae->latent_t +
                        (size_t)source_t) *
                        (size_t)vae->latent_h + (size_t)height) *
                        (size_t)vae->latent_w + (size_t)width;
                    rows[row++] = input[source] * deviation[channel] +
                                  mean[channel];
                }
    }
    vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows, patch_elements);
    free(rows);
    return vae->latent != NULL;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    size_t count = (size_t)vae->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating video VAE RoPE");
        return 0;
    }
    uint32_t row = 0;
    for (int t = 0; t < CHUNK_LATENT_TIME; t++)
        for (int h = 0; h < vae->latent_h; h++)
            for (int w = 0; w < vae->latent_w; w++, row++) {
                float axes[] = {
                    2.0f * (((float)t + 0.5f) /
                            (float)CHUNK_LATENT_TIME) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
                };
                for (uint32_t axis = 0; axis < 3; axis++)
                    for (uint32_t frequency = 0; frequency < 8; frequency++) {
                        float inverse = 1.0f /
                            powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = (size_t)row * ROPE_HALF +
                                        axis * 8 + frequency;
                        cosines[offset] = cosf(angle);
                        sines[offset] = sinf(angle);
                    }
            }
    while (row < vae->sequence) {
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
        row++;
    }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    free(cosines); free(sines);
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate video VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    size_t patches = vae->patches, sequence = vae->sequence;
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * HIDDEN),
        F32(hidden, sequence * HIDDEN),
        F32(norm, sequence * HIDDEN),
        F32(qkv, sequence * INNER * 3),
        F32(query, sequence * INNER), F32(key, sequence * INNER),
        F32(value, sequence * INNER), F32(heads, sequence * INNER),
        F32(branch, sequence * HIDDEN), F32(ff1, sequence * FFN * 2),
        F32(activated, sequence * FFN),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate video VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    if (vae->precision == VAE_F32) return 1;
    size_t padded = (sequence + 127u) & ~(size_t)127u;
    vae->lowp_in = h3_gpu_tensor_new_bf16(vae->gpu, sequence * FFN);
    vae->lowp_out = h3_gpu_tensor_new_bf16(vae->gpu, sequence * FFN * 2);
    if (vae->precision == VAE_INT8) {
        vae->quant_in = h3_gpu_tensor_new_i8(vae->gpu, padded * FFN);
        vae->quant_scales = h3_gpu_tensor_new_f32(vae->gpu, padded);
    }
    if (vae->bf16_attention) {
        vae->query_bf16 = h3_gpu_tensor_new_bf16(vae->gpu, sequence * INNER);
        vae->key_bf16 = h3_gpu_tensor_new_bf16(vae->gpu, sequence * INNER);
        vae->value_bf16 = h3_gpu_tensor_new_bf16(vae->gpu, sequence * INNER);
        if (!vae->query_bf16 || !vae->key_bf16 || !vae->value_bf16) {
            fail(error, error_size, "cannot allocate video VAE BF16 "
                 "attention: %s", h3_gpu_error(vae->gpu));
            return 0;
        }
    }
    if (!vae->lowp_in || !vae->lowp_out ||
        (vae->precision == VAE_INT8 &&
         (!vae->quant_in || !vae->quant_scales))) {
        fail(error, error_size, "cannot allocate video VAE low-precision "
             "activations: %s", h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

/* lowp_out (BF16) = lowp_in (BF16) @ weight^T through the lowered weight. */
static int lowp_matmul(vae_context *vae, const vae_block *block, int which,
                       uint32_t input_dim, uint32_t output_dim) {
    uint32_t rows = vae->sequence;
    return vae->precision == VAE_INT8 ?
        h3_gpu_linear_int8_bf16(vae->gpu, vae->lowp_out, vae->quant_in,
                                vae->quant_scales, vae->lowp_in,
                                block->lowp[which],
                                block->lowp_scales[which], rows, input_dim,
                                output_dim, 0) :
        h3_gpu_linear_bf16(vae->gpu, vae->lowp_out, vae->lowp_in,
                           block->lowp[which], NULL, rows, input_dim,
                           output_dim);
}

/* output_f32 = input_f32 @ weight^T + bias through the lowered weight. */
static int lowp_linear(vae_context *vae, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input, const vae_block *block,
                       int which, const h3_gpu_tensor *bias,
                       uint32_t input_dim, uint32_t output_dim) {
    uint32_t rows = vae->sequence;
    /* input == NULL: the caller already left the BF16 rows in lowp_in. */
    if (input && !h3_gpu_cast_f32_to_bf16(vae->gpu, vae->lowp_in, input,
                                          rows * input_dim)) return 0;
    return lowp_matmul(vae, block, which, input_dim, output_dim) &&
           h3_gpu_bias_bf16_to_f32(vae->gpu, output, vae->lowp_out,
                                   bias, rows, output_dim);
}

/* The low-precision block of run_block with its BF16<->F32 epilogues fused:
 * the norms write BF16 matmul inputs directly, the QKV bias, QK norm, RoPE
 * and BF16 attention casts run as one kernel, SwiGLU consumes the biased BF16
 * matmul output, and the output/MLP biases fold into the residual updates.
 * Every value is computed exactly as in run_block (see h3_shaders.metal), so
 * the block output is bit-identical; F32 qkv/ff1/activated/branch traffic and
 * ten of the sixteen element-wise dispatches per block go away. Needs BF16
 * attention. */
static int run_block_fused(vae_context *vae, int index, char *error,
                           size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_f32_to_bf16(vae->gpu, vae->lowp_in, vae->hidden,
        weight->norm1, rows, HIDDEN, 1e-5f), "video VAE attention norm");
    OP(lowp_matmul(vae, weight, LP_QKV, HIDDEN, INNER * 3), "video VAE QKV");
    OP(h3_gpu_video_qkv_bias_rope_bf16(vae->gpu, vae->query_bf16,
        vae->key_bf16, vae->value_bf16, vae->lowp_out, weight->qkv_b,
        vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM, ROPE_HALF,
        1e-5f), "video VAE QK norm/RoPE");
    OP(h3_gpu_sdpa_bf16(vae->gpu, vae->lowp_in, vae->query_bf16,
        vae->key_bf16, vae->value_bf16, rows, HEADS, HEAD_DIM,
        1.0f / sqrtf((float)HEAD_DIM)), "video VAE BF16 attention");
    OP(lowp_matmul(vae, weight, LP_OUT, INNER, HIDDEN),
       "video VAE attention output");
    OP(h3_gpu_bias_scale_add_bf16(vae->gpu, vae->hidden, vae->hidden,
        vae->lowp_out, weight->out_b, weight->scale1, rows, HIDDEN),
       "video VAE attention residual");
    OP(h3_gpu_rms_norm_f32_to_bf16(vae->gpu, vae->lowp_in, vae->hidden,
        weight->norm2, rows, HIDDEN, 1e-5f), "video VAE MLP norm");
    OP(lowp_matmul(vae, weight, LP_W1, HIDDEN, FFN * 2),
       "video VAE MLP input");
    OP(h3_gpu_bias_swiglu_bf16(vae->gpu, vae->lowp_in, vae->lowp_out,
        weight->w1_b, rows, FFN), "video VAE SwiGLU");
    OP(lowp_matmul(vae, weight, LP_W2, FFN, HIDDEN), "video VAE MLP output");
    OP(h3_gpu_bias_scale_add_bf16(vae->gpu, vae->hidden, vae->hidden,
        vae->lowp_out, weight->w2_b, weight->scale2, rows, HIDDEN),
       "video VAE MLP residual");
#undef OP
    return 1;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
    if (vae->fused) return run_block_fused(vae, index, error, error_size);
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "video VAE attention norm");
    OP(vae->precision ?
        lowp_linear(vae, vae->qkv, vae->norm, weight, LP_QKV, weight->qkv_b,
                    HIDDEN, INNER * 3) :
        h3_gpu_linear_f32(vae->gpu, vae->qkv, vae->norm, weight->qkv_w,
        weight->qkv_b, rows, HIDDEN, INNER * 3), "video VAE QKV");
    OP(h3_gpu_video_qkv_rope_f32(vae->gpu, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
    if (vae->bf16_attention) {
        uint32_t elements = rows * INNER;
        OP(h3_gpu_cast_f32_to_bf16(vae->gpu, vae->query_bf16, vae->query,
            elements) &&
           h3_gpu_cast_f32_to_bf16(vae->gpu, vae->key_bf16, vae->key,
            elements) &&
           h3_gpu_cast_f32_to_bf16(vae->gpu, vae->value_bf16, vae->value,
            elements), "video VAE BF16 attention inputs");
        OP(h3_gpu_sdpa_bf16(vae->gpu, vae->lowp_in, vae->query_bf16,
            vae->key_bf16, vae->value_bf16, rows, HEADS, HEAD_DIM,
            1.0f / sqrtf((float)HEAD_DIM)), "video VAE BF16 attention");
    } else
    OP(h3_gpu_sdpa_f32(vae->gpu, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "video VAE attention");
    OP(vae->precision ?
        lowp_linear(vae, vae->branch,
                    vae->bf16_attention ? NULL : vae->heads, weight, LP_OUT,
                    weight->out_b, INNER, HIDDEN) :
        h3_gpu_linear_f32(vae->gpu, vae->branch, vae->heads, weight->out_w,
        weight->out_b, rows, INNER, HIDDEN), "video VAE attention output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "video VAE attention residual");
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm2,
        rows, HIDDEN, 1e-5f), "video VAE MLP norm");
    OP(vae->precision ?
        lowp_linear(vae, vae->ff1, vae->norm, weight, LP_W1, weight->w1_b,
                    HIDDEN, FFN * 2) :
        h3_gpu_linear_f32(vae->gpu, vae->ff1, vae->norm, weight->w1,
        weight->w1_b, rows, HIDDEN, FFN * 2), "video VAE MLP input");
    OP(h3_gpu_swiglu_f32(vae->gpu, vae->activated, vae->ff1, rows, FFN),
       "video VAE SwiGLU");
    OP(vae->precision ?
        lowp_linear(vae, vae->branch, vae->activated, weight, LP_W2,
                    weight->w2_b, FFN, HIDDEN) :
        h3_gpu_linear_f32(vae->gpu, vae->branch, vae->activated, weight->w2,
        weight->w2_b, rows, FFN, HIDDEN), "video VAE MLP output");
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "video VAE MLP residual");
#undef OP
    return 1;
}

static int run_decoder(vae_context *vae, h3_video_vae_progress progress,
                       void *progress_opaque, char *error,
                       size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin video VAE decoder");
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack video VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers);

    for (int index = 0; index < LAYERS; index++) {
        if (!load_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_begin(vae->gpu), "begin video VAE transformer block");
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit video VAE transformer block");
        free_block(&vae->blocks[index]);
        if (progress) progress(index + 1, LAYERS, progress_opaque);
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    OP(h3_gpu_begin(vae->gpu), "begin video VAE output projection");
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "video VAE output projection");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE decoder");
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
#undef OP
    return 1;
}

static int cache_is_valid(const char *path, const uint8_t fingerprint[32]) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    vae_cache_header header;
    int ok = fread(&header, sizeof(header), 1, file) == 1 &&
             !memcmp(header.magic, VAE_CACHE_MAGIC, sizeof(header.magic)) &&
             header.layers == LAYERS &&
             !memcmp(header.fingerprint, fingerprint, 32) &&
             !fseeko(file, 0, SEEK_END) &&
             ftello(file) == (off_t)(sizeof(header) +
                                     (size_t)LAYERS * cache_block_bytes());
    fclose(file);
    return ok;
}

static char *cache_temp_path(const char *path) {
    size_t size = strlen(path) + 32;
    char *temp = malloc(size);
    if (temp) snprintf(temp, size, "%s.tmp.%d", path, (int)getpid());
    return temp;
}

static int is_safetensors(const struct dirent *entry) {
    size_t length = strlen(entry->d_name);
    return length > 12 && !strcmp(entry->d_name + length - 12,
                                  ".safetensors");
}

/* Cheap staleness check, not a content hash: FNV-1a over the resolved
 * directory and each shard's name, size and mtime. */
static void weight_fingerprint(const char *directory, uint8_t out[32]) {
    uint64_t hash = 1469598103934665603ULL;
#define MIX(data, length) do {                                                  \
    const uint8_t *bytes = (const uint8_t *)(data);                             \
    for (size_t at = 0; at < (size_t)(length); at++) {                          \
        hash ^= bytes[at];                                                      \
        hash *= 1099511628211ULL;                                               \
    }                                                                           \
} while (0)
    MIX(directory, strlen(directory));
    struct dirent **entries = NULL;
    int count = scandir(directory, &entries, is_safetensors, alphasort);
    for (int index = 0; index < count; index++) {
        char path[2048];
        struct stat info;
        snprintf(path, sizeof(path), "%s/%s", directory,
                 entries[index]->d_name);
        if (!stat(path, &info)) {
            int64_t fields[] = {(int64_t)info.st_size,
                                (int64_t)info.st_mtimespec.tv_sec,
                                (int64_t)info.st_mtimespec.tv_nsec};
            MIX(entries[index]->d_name, strlen(entries[index]->d_name));
            MIX(fields, sizeof(fields));
        }
        free(entries[index]);
    }
    free(entries);
#undef MIX
    memset(out, 0, 32);
    memcpy(out, &hash, sizeof(hash));
}

/* Open through the resolved path so linked model directories (for example a
 * folded-LoRA checkpoint linking back to video_vae) share one VAE cache. */
static h3_weight_store *open_weights(vae_context *vae, const char *directory,
                                     char *error, size_t error_size) {
    char *resolved = realpath(directory, NULL);
    snprintf(vae->weight_path, sizeof(vae->weight_path), "%s",
             resolved ? resolved : directory);
    free(resolved);
    return h3_weight_store_open(vae->weight_path, error, error_size);
}

/* Choose between reading a valid cache and writing a fresh one. */
static void open_vae_cache(vae_context *vae, char **temp_path) {
    const char *directory = getenv("H3_VAE_CACHE");
    *temp_path = NULL;
    if (!directory || !*directory || vae->precision != VAE_BF16) return;
    vae_cache_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, VAE_CACHE_MAGIC, sizeof(header.magic));
    header.layers = LAYERS;
    weight_fingerprint(vae->weight_path, header.fingerprint);
    /* One file per weight set, named by its fingerprint. */
    snprintf(vae->cache_file, sizeof(vae->cache_file),
             "%s/video-vae-bf16-%02x%02x%02x%02x%02x%02x%02x%02x.cache",
             directory, header.fingerprint[0], header.fingerprint[1],
             header.fingerprint[2], header.fingerprint[3],
             header.fingerprint[4], header.fingerprint[5],
             header.fingerprint[6], header.fingerprint[7]);
    const char *path = vae->cache_file;
    if (cache_is_valid(path, header.fingerprint)) {
        vae->cache_path = path;
        fprintf(stderr, "h3: video VAE BF16 cache hit %s\n", path);
        return;
    }
    *temp_path = cache_temp_path(path);
    vae->cache_host = malloc(LOWP_ELEMENTS[LP_W1] * sizeof(uint16_t));
    vae->cache_out = *temp_path && vae->cache_host ?
        fopen(*temp_path, "wb") : NULL;
    if (vae->cache_out &&
        fwrite(&header, sizeof(header), 1, vae->cache_out) != 1) {
        fclose(vae->cache_out);
        vae->cache_out = NULL;
    }
    if (!vae->cache_out)
        fprintf(stderr, "h3: warning: cannot create video VAE cache %s\n",
                path);
}

static void close_vae_cache(vae_context *vae, char *temp_path, int ok) {
    const char *path = vae->cache_file;
    if (vae->cache_out) {
        int written = !fclose(vae->cache_out) && ok;
        vae->cache_out = NULL;
        if (written && !rename(temp_path, path))
            fprintf(stderr, "h3: video VAE BF16 cache written %s\n", path);
        else
            fprintf(stderr, "h3: warning: video VAE cache not written\n");
    }
    if (temp_path) unlink(temp_path);
    free(temp_path);
    free(vae->cache_host);
    vae->cache_host = NULL;
    vae->cache_path = NULL;
}

static int load_resident_weights(vae_context *vae,
                                 h3_video_vae_progress progress,
                                 void *progress_opaque, char *error,
                                 size_t error_size) {
    char *temp_path = NULL;
    open_vae_cache(vae, &temp_path);
    int ok = load_input_weights(vae, error, error_size);
    for (int index = 0; ok && index < LAYERS; index++) {
        ok = load_block(vae, index, error, error_size);
        if (ok && progress) progress(index + 1, LAYERS, progress_opaque);
    }
    ok = ok && load_output_weights(vae, error, error_size);
    close_vae_cache(vae, temp_path, ok);
    return ok;
}

/* Encode one resident tile, from vae->latent to vae->projected, into the open
 * command buffer. zero is the [HIDDEN] F32 zero suffix token. */
static int encode_resident_tile(vae_context *vae, const h3_gpu_tensor *zero,
                                char *error, size_t error_size) {
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
    OP(h3_gpu_linear_f32(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "tiled video VAE post-quant projection");
    OP(h3_gpu_linear_f32(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "tiled video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack tiled video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack tiled video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack tiled video VAE suffix");
    for (int index = 0; index < LAYERS; index++)
        if (!run_block(vae, index, error, error_size)) return 0;
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "tiled video VAE output LayerNorm");
    OP(h3_gpu_linear_f32(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "tiled video VAE output projection");
#undef OP
    return 1;
}

/* Tiled decoding keeps every VAE weight resident for the duration of the phase.
 * This uses about 9 GiB after the DiT has been retired, avoids rereading 9 GiB
 * per spatial tile, and permits one command-buffer chain per tile. */
static int run_resident_tile(vae_context *vae, char *error,
                             size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
    int ok = gpu_op(vae, h3_gpu_begin(vae->gpu), error, error_size,
                    "begin tiled video VAE decoder") &&
             encode_resident_tile(vae, zero, error, error_size) &&
             gpu_op(vae, h3_gpu_submit(vae->gpu), error, error_size,
                    "submit tiled video VAE decoder");
    h3_gpu_tensor_free(zero);
    return ok;
}

/* Projected patch rows [sequence, 3*4*16*16] -> RGB frames [frame, y, x, 3]
 * in [0,1] for frames [first_frame, first_frame + frame_count). */
static void unpack_patch_rows(const float *rows, int latent_h, int latent_w,
                              int output_frames, int first_frame,
                              int frame_count, float *rgb) {
    int pixel_h = latent_h * 16;
    int pixel_w = latent_w * 16;
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    for (int output_frame = 0; output_frame < frame_count; output_frame++) {
        int frame = first_frame + output_frame;
        int decoded_t = frame + FRAME_OFFSET;
        if (output_frames == FIRST_CHUNK_FRAMES && frame >= 17)
            decoded_t += 3;
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        for (int y = 0; y < pixel_h; y++) {
            int patch_y = y / 16, within_y = y % 16;
            for (int x = 0; x < pixel_w; x++) {
                int patch_x = x / 16, within_x = x % 16;
                size_t patch = ((size_t)patch_t * (size_t)latent_h +
                    (size_t)patch_y) * (size_t)latent_w +
                    (size_t)patch_x;
                for (int channel = 0; channel < 3; channel++) {
                    size_t component = ((((size_t)channel * 4 +
                        (size_t)within_t) * 16 + (size_t)within_y) * 16 +
                        (size_t)within_x);
                    float value = rows[patch * OUTPUT_PATCH + component] *
                                  deviation[channel] + mean[channel];
                    value = fminf(fmaxf(value, 0.0f), 1.0f);
                    size_t destination = (((size_t)output_frame *
                        (size_t)pixel_h +
                        (size_t)y) * (size_t)pixel_w + (size_t)x) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
            }
        }
    }
}

static int unpack_frame_range(vae_context *vae, int first_frame,
                              int frame_count, h3_video_frames *output,
                              char *error, size_t error_size) {
    if (first_frame < 0 || frame_count < 1 ||
        first_frame > vae->output_frames - frame_count) {
        fail(error, error_size, "invalid video VAE output frame range");
        return 0;
    }
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = malloc(projected_count * sizeof(*rows));
    int pixel_h = vae->latent_h * 16;
    int pixel_w = vae->latent_w * 16;
    size_t rgb_count = (size_t)frame_count * (size_t)pixel_h *
                       (size_t)pixel_w * 3;
    float *rgb = malloc(rgb_count * sizeof(*rgb));
    if (!rows || !rgb) {
        free(rows); free(rgb);
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32(vae->projected, rows, projected_count)) {
        free(rows); free(rgb);
        fail(error, error_size, "cannot read video VAE output patches");
        return 0;
    }
    unpack_patch_rows(rows, vae->latent_h, vae->latent_w, vae->output_frames,
                      first_frame, frame_count, rgb);
    free(rows);
    output->frames = frame_count;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    return h3_gpu_get_stats(vae->gpu, &output->gpu_stats);
}

static int unpack_frames(vae_context *vae, h3_video_frames *output,
                         char *error, size_t error_size) {
    return unpack_frame_range(vae, 0, vae->output_frames, output,
                              error, error_size);
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

static int configured_tile_pixels(int pixel_height, int pixel_width) {
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    if (value && *value) {
        char *end = NULL;
        long pixels = strtol(value, &end, 10);
        if (end && !*end && pixels >= TILE_PIXELS && pixels <= 512 &&
            pixels % SPATIAL_RATIO == 0) return (int)pixels;
    }
    int best = TILE_PIXELS;
    uint64_t best_score = UINT64_MAX;
    for (int pixels = TILE_PIXELS; pixels <= 320;
         pixels += SPATIAL_RATIO) {
        uint64_t tiles = (uint64_t)tile_count_for_extent(pixel_height, pixels) *
            (uint64_t)tile_count_for_extent(pixel_width, pixels);
        /* The resident VAE is dominated by linears and activation traffic;
         * measured tile cost follows area more closely than cubic sequence
         * growth at these shapes. Attention is still included in each tile. */
        uint64_t score = tiles * (uint64_t)pixels * (uint64_t)pixels;
        if (score < best_score) {
            best = pixels;
            best_score = score;
        }
    }
    return best;
}

static int tile_axis_build(int extent, int tile_pixels, tile_axis *axis,
                           char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % SPATIAL_RATIO) {
        fail(error, error_size, "video VAE tile extent must be a multiple of %d",
             SPATIAL_RATIO);
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing VAE tile plan");
            return 0;
        }
        return 1;
    }
    int count = tile_count_for_extent(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing VAE tile plan");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

static float *extract_latent_tile(const float *latent, int full_t,
                                  int full_h, int full_w, int start_t,
                                  int start_y, int start_x, int tile_t,
                                  int tile_h, int tile_w, char *error,
                                  size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_t *
                   (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting video VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int time = 0; time < tile_t; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * (size_t)full_t +
                    (size_t)(start_t + time)) * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)start_x;
                size_t destination = (((size_t)channel * (size_t)tile_t +
                    (size_t)time) * (size_t)tile_h + (size_t)y) *
                    (size_t)tile_w;
                memcpy(tile + destination, latent + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

/* Blend tile (tile_y, tile_x) into the stitched frames rgb [frame, full_h,
 * full_w, 3]: its top and left overlaps ramp linearly from the tile above
 * and the tile to the left, which therefore must already be decoded. */
static void stitch_tile(const float *const *tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int tile_y, int tile_x,
                        int frame_count, int full_h, int full_w, float *rgb) {
    int index = tile_y * x_axis->count + tile_x;
    const float *current = tiles[index];
    const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
    const float *left = tile_x ? tiles[index - 1] : NULL;
    int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
    int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
    int keep_h = y_axis->length -
        (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
    int keep_w = x_axis->length -
        (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
    for (int frame = 0; frame < frame_count; frame++)
        for (int y = 0; y < keep_h; y++)
            for (int x = 0; x < keep_w; x++)
                for (int channel = 0; channel < 3; channel++) {
                    size_t local = (((size_t)frame *
                        (size_t)y_axis->length + (size_t)y) *
                        (size_t)x_axis->length + (size_t)x) * 3 +
                        (size_t)channel;
                    float value = current[local];
                    if (above && y < overlap_y) {
                        size_t top = (((size_t)frame *
                            (size_t)y_axis->length +
                            (size_t)(y_axis->length - overlap_y + y)) *
                            (size_t)x_axis->length + (size_t)x) * 3 +
                            (size_t)channel;
                        float alpha = (float)y / (float)overlap_y;
                        value = above[top] * (1.0f - alpha) +
                                value * alpha;
                    }
                    if (left && x < overlap_x) {
                        size_t prior = (((size_t)frame *
                            (size_t)y_axis->length + (size_t)y) *
                            (size_t)x_axis->length +
                            (size_t)(x_axis->length - overlap_x + x)) *
                            3 + (size_t)channel;
                        float alpha = (float)x / (float)overlap_x;
                        value = left[prior] * (1.0f - alpha) +
                                value * alpha;
                    }
                    size_t destination = (((size_t)frame *
                        (size_t)full_h +
                        (size_t)(y_axis->starts[tile_y] + y)) *
                        (size_t)full_w +
                        (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
}

static int stitch_tiles(float **tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h *
                   (size_t)full_w * 3;
    float *rgb = malloc(count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory stitching video VAE tiles");
        return 0;
    }
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++)
            stitch_tile((const float *const *)tiles, y_axis, x_axis, tile_y,
                        tile_x, frame_count, full_h, full_w, rgb);
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    return 1;
}

/* Chunks after the first ramp their five leading frames in from the five
 * trailing frames of the previous chunk. */
static void blend_temporal_overlap(float *decoded, const float *overlap,
                                   size_t frame_elements) {
    for (int frame = 0; frame < 5; frame++) {
        float alpha = (float)frame / 5.0f;
        size_t base = (size_t)frame * frame_elements;
        for (size_t index = 0; index < frame_elements; index++)
            decoded[base + index] = overlap[base + index] *
                (1.0f - alpha) + decoded[base + index] * alpha;
    }
}

static double vae_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && *value && strcmp(value, "0");
}

/* Host half of the pipelined decode. Each finished tile is unpacked into a
 * reused buffer and stitched into the current chunk at once (tiles finish in
 * raster order, so the tiles above and to the left are always present); when
 * a chunk's last tile lands, its temporal blend runs and its 17 final frames
 * go to the sink. Every value goes through the same unpack_patch_rows,
 * stitch_tile and blend_temporal_overlap arithmetic as the legacy decode. */
typedef struct {
    const tile_axis *y_axis;
    const tile_axis *x_axis;
    int latent_h;               /* per-tile latent extent */
    int latent_w;
    int tile_count;
    int chunks;
    int full_h;
    int full_w;
    size_t frame_elements;
    float *tile_block;
    float **tiles;
    float *chunk_rgb;           /* FIRST_CHUNK_FRAMES stitched frames */
    float *overlap;             /* the 5 trailing frames of the last chunk */
    h3_video_frame_sink sink;
    void *sink_opaque;
    int profile;
    double started;
    double unpack_seconds;
    double stitch_seconds;
    double blend_seconds;
    double sink_seconds;
} vae_assembler;

static void assembler_free(vae_assembler *assembler) {
    free(assembler->tile_block);
    free(assembler->tiles);
    free(assembler->chunk_rgb);
    free(assembler->overlap);
    memset(assembler, 0, sizeof(*assembler));
}

static int assembler_init(vae_assembler *assembler, const tile_axis *y_axis,
                          const tile_axis *x_axis, int chunks,
                          h3_video_frame_sink sink, void *sink_opaque,
                          char *error, size_t error_size) {
    memset(assembler, 0, sizeof(*assembler));
    assembler->y_axis = y_axis;
    assembler->x_axis = x_axis;
    assembler->latent_h = y_axis->length / SPATIAL_RATIO;
    assembler->latent_w = x_axis->length / SPATIAL_RATIO;
    assembler->tile_count = y_axis->count * x_axis->count;
    assembler->chunks = chunks;
    assembler->full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    assembler->full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    assembler->frame_elements = (size_t)assembler->full_h *
                                (size_t)assembler->full_w * 3;
    assembler->sink = sink;
    assembler->sink_opaque = sink_opaque;
    assembler->profile = env_enabled("H3_PROFILE");
    assembler->started = vae_seconds();
    size_t tile_elements = (size_t)FIRST_CHUNK_FRAMES *
                           (size_t)y_axis->length * (size_t)x_axis->length * 3;
    assembler->tile_block = malloc((size_t)assembler->tile_count *
                                   tile_elements * sizeof(float));
    assembler->tiles = calloc((size_t)assembler->tile_count,
                              sizeof(*assembler->tiles));
    assembler->chunk_rgb = malloc((size_t)FIRST_CHUNK_FRAMES *
                                  assembler->frame_elements * sizeof(float));
    assembler->overlap = malloc(5 * assembler->frame_elements *
                                sizeof(float));
    if (!assembler->tile_block || !assembler->tiles ||
        !assembler->chunk_rgb || !assembler->overlap) {
        assembler_free(assembler);
        fail(error, error_size, "out of memory assembling video VAE frames");
        return 0;
    }
    for (int index = 0; index < assembler->tile_count; index++)
        assembler->tiles[index] = assembler->tile_block +
                                  (size_t)index * tile_elements;
    return 1;
}

static int assembler_deliver(vae_assembler *assembler, const float *rgb,
                             int first, int count, char *error,
                             size_t error_size) {
    double started = vae_seconds();
    int ok = assembler->sink(rgb, first, count, assembler->full_w,
                             assembler->full_h, assembler->sink_opaque);
    double finished = vae_seconds();
    assembler->sink_seconds += finished - started;
    if (assembler->profile)
        fprintf(stderr, "h3 profile: video VAE frames %3d-%3d delivered "
                "at %7.3fs (sink %.3fs)\n", first, first + count - 1,
                finished - assembler->started, finished - started);
    if (!ok) fail(error, error_size, "video frame consumer stopped the decode");
    return ok;
}

/* Accept the projected patch rows of job = chunk * tile_count + tile. */
static int assembler_take(vae_assembler *assembler, int job,
                          const float *rows, char *error, size_t error_size) {
    int chunk = job / assembler->tile_count;
    int tile = job % assembler->tile_count;
    double started = vae_seconds();
    unpack_patch_rows(rows, assembler->latent_h, assembler->latent_w,
                      FIRST_CHUNK_FRAMES, 0, FIRST_CHUNK_FRAMES,
                      assembler->tiles[tile]);
    double unpacked = vae_seconds();
    stitch_tile((const float *const *)assembler->tiles, assembler->y_axis,
                assembler->x_axis, tile / assembler->x_axis->count,
                tile % assembler->x_axis->count, FIRST_CHUNK_FRAMES,
                assembler->full_h, assembler->full_w, assembler->chunk_rgb);
    double stitched = vae_seconds();
    assembler->unpack_seconds += unpacked - started;
    assembler->stitch_seconds += stitched - unpacked;
    if (tile + 1 < assembler->tile_count) return 1;
    if (chunk) blend_temporal_overlap(assembler->chunk_rgb, assembler->overlap,
                                      assembler->frame_elements);
    assembler->blend_seconds += vae_seconds() - stitched;
    if (!assembler_deliver(assembler, assembler->chunk_rgb, chunk * 17, 17,
                           error, error_size)) return 0;
    memcpy(assembler->overlap,
           assembler->chunk_rgb + 17 * assembler->frame_elements,
           5 * assembler->frame_elements * sizeof(float));
    return 1;
}

static int assembler_finish(vae_assembler *assembler, char *error,
                            size_t error_size) {
    return assembler_deliver(assembler, assembler->overlap,
                             assembler->chunks * 17, 5, error, error_size);
}

typedef struct {
    float *rgb;
    size_t frame_elements;
} frame_copy;

static int copy_frames(const float *rgb, int first, int count, int width,
                       int height, void *opaque) {
    (void)width;
    (void)height;
    frame_copy *copy = opaque;
    memcpy(copy->rgb + (size_t)first * copy->frame_elements, rgb,
           (size_t)count * copy->frame_elements * sizeof(float));
    return 1;
}

static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    int frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    int ok = 1;
    for (int tile_y = 0; tile_y < decoder->y_axis.count && ok; tile_y++)
        for (int tile_x = 0; tile_x < decoder->x_axis.count && ok; tile_x++) {
            float *input = extract_latent_tile(
                normalized_latent, latent_time, decoder->latent_h,
                decoder->latent_w, chunk * 5,
                decoder->y_axis.starts[tile_y] / SPATIAL_RATIO,
                decoder->x_axis.starts[tile_x] / SPATIAL_RATIO,
                CHUNK_LATENT_TIME, decoder->vae.latent_h,
                decoder->vae.latent_w, error, error_size);
            if (!input) {
                ok = 0;
                break;
            }
            free_tensor(&decoder->vae.latent);
            ok = prepare_input(&decoder->vae, input,
                               decoder->latent_mean, decoder->latent_std,
                               error, error_size) &&
                 run_resident_tile(&decoder->vae, error, error_size);
            free(input);
            if (!ok) break;
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(&decoder->vae, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(&decoder->vae, &tile, error, error_size);
            if (ok) {
                int index = tile_y * decoder->x_axis.count + tile_x;
                tiles[index] = tile.rgb;
            }
        }
    if (ok) ok = stitch_tiles(tiles, &decoder->y_axis, &decoder->x_axis,
                              frame_count, output, error, error_size);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE arguments");
        return NULL;
    }
    h3_video_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating resident video VAE");
        return NULL;
    }
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    int ok = load_latent_normalization(
        weight_directory, decoder->latent_mean, decoder->latent_std,
        error, error_size) &&
        tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                        &decoder->y_axis, error, error_size) &&
        tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                        &decoder->x_axis, error, error_size);
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    vae_context *vae = &decoder->vae;
    if (ok) {
        vae->latent_h = decoder->y_axis.length / SPATIAL_RATIO;
        vae->latent_w = decoder->x_axis.length / SPATIAL_RATIO;
        vae->latent_t = CHUNK_LATENT_TIME;
        vae->output_frames = FIRST_CHUNK_FRAMES;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * vae->latent_h *
                                  vae->latent_w);
        vae->sequence = vae->patches + SUFFIX;
        vae->weights = open_weights(vae, weight_directory, error,
                                    error_size);
        if (vae->weights)
            vae->gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (vae->gpu) vae->precision = configured_precision(vae->gpu);
        if (vae->gpu) vae->bf16_attention = vae->precision != VAE_F32 &&
            !getenv("H3_VAE_F32_ATTENTION");
        if (vae->gpu) vae->fused = vae->bf16_attention &&
            h3_video_vae_pipelined() && !env_enabled("H3_VAE_UNFUSED");
        if (vae->gpu)
            h3_gpu_profile_set_label(vae->gpu, "resident video VAE decoder");
        ok = vae->weights && vae->gpu &&
             load_resident_weights(vae, progress, progress_opaque,
                                   error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size);
    }
    if (!ok) {
        h3_video_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || !output_frame_index ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid video VAE preview arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int local_frame = FIRST_CHUNK_FRAMES / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + local_frame;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    int ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, local_frame, output,
                                  error, error_size);
    if (ok) *output_frame_index = global_frame;
    return ok;
}

int h3_video_vae_pipelined(void) {
    return !env_enabled("H3_VAE_DECODE_LEGACY");
}

/* The released resident decode: one blocking submit per tile, then a host
 * unpack into fresh buffers, a stitch per chunk and one output buffer. */
static int decoder_decode_legacy(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    double started = vae_seconds();
    h3_gpu_stats before;
    memset(&before, 0, sizeof(before));
    h3_gpu_get_stats(decoder->vae.gpu, &before);
    float *final_rgb = malloc(output_elements * sizeof(*final_rgb));
    float *temporal_overlap = malloc(5 * frame_elements *
                                     sizeof(*temporal_overlap));
    if (!final_rgb || !temporal_overlap) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = 1;
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, -1, &decoded, error, error_size);
        if (!ok) break;
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (!ok) h3_video_frames_free(output);
    if (ok && env_enabled("H3_PROFILE")) {
        double wall = vae_seconds() - started;
        double wait = output->gpu_stats.command_wait_seconds -
                      before.command_wait_seconds;
        fprintf(stderr, "h3 profile: video VAE decode legacy fused=%d "
                "tiles=%d chunks=%d wall=%.3fs gpu-wait=%.3fs "
                "host+encode=%.3fs\n", decoder->vae.fused,
                decoder->y_axis.count * decoder->x_axis.count, chunks,
                wall, wait, wall - wait);
    }
    return ok;
}

static int ensure_pipeline_slots(h3_video_vae_decoder *decoder, char *error,
                                 size_t error_size) {
    vae_context *vae = &decoder->vae;
    size_t latent_elements = (size_t)vae->patches * LATENT_CHANNELS;
    for (int slot = 0; slot < 2; slot++)
        if (!decoder->slot_latent[slot])
            decoder->slot_latent[slot] =
                h3_gpu_tensor_new_f32(vae->gpu, latent_elements);
    if (!decoder->slot_projected)
        decoder->slot_projected = h3_gpu_tensor_new_f32(
            vae->gpu, (size_t)vae->sequence * OUTPUT_PATCH);
    if (!decoder->suffix_zero) {
        float zeros[HIDDEN];
        memset(zeros, 0, sizeof(zeros));
        decoder->suffix_zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    }
    if (decoder->slot_latent[0] && decoder->slot_latent[1] &&
        decoder->slot_projected && decoder->suffix_zero) return 1;
    fail(error, error_size, "cannot allocate pipelined video VAE slots: %s",
         h3_gpu_error(vae->gpu));
    return 0;
}

/* extract_latent_tile followed by prepare_input's denormalization, written
 * straight into the tile's patch rows. */
static void fill_tile_input(const h3_video_vae_decoder *decoder,
                            const float *latent, int latent_time, int chunk,
                            int tile_y, int tile_x, float *rows) {
    const vae_context *vae = &decoder->vae;
    int start_t = chunk * 5;
    int start_y = decoder->y_axis.starts[tile_y] / SPATIAL_RATIO;
    int start_x = decoder->x_axis.starts[tile_x] / SPATIAL_RATIO;
    const float *mean = decoder->latent_mean;
    const float *deviation = decoder->latent_std;
    size_t row = 0;
    for (int time = 0; time < CHUNK_LATENT_TIME; time++)
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)latent_time +
                        (size_t)(start_t + time)) *
                        (size_t)decoder->latent_h +
                        (size_t)(start_y + height)) *
                        (size_t)decoder->latent_w + (size_t)(start_x + width);
                    rows[row++] = latent[source] * deviation[channel] +
                                  mean[channel];
                }
}

/* Retire whatever is still queued after a failure so no buffer is freed or
 * rewritten while the GPU uses it. */
static void drain_gpu_chain(h3_gpu *gpu) {
    if (h3_gpu_submit(gpu)) return;
    for (int index = 0; index < 3; index++) h3_gpu_wait_oldest(gpu);
}

/* Same tiles, kernels and host arithmetic as decoder_decode_legacy, but the
 * GPU always has the next tile queued: tile N+1 is committed before the host
 * waits for tile N, then unpacks and stitches N while N+1 runs. Two latent
 * and two output slots alternate, so a slot is rewritten only after the host
 * has consumed the tile that last used it; the queue orders everything else.
 * Frames reach the sink as each chunk completes. */
static int decoder_decode_pipelined(h3_video_vae_decoder *decoder,
                                    const float *latent, int latent_time,
                                    h3_video_frame_sink sink, void *opaque,
                                    char *error, size_t error_size) {
    vae_context *vae = &decoder->vae;
    int chunks = (latent_time - 2) / 5;
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    int jobs = chunks * tile_count;
    size_t latent_elements = (size_t)vae->patches * LATENT_CHANNELS;
    double started = vae_seconds();
    h3_gpu_stats before;
    memset(&before, 0, sizeof(before));
    h3_gpu_get_stats(vae->gpu, &before);
    float *input_rows = malloc(latent_elements * sizeof(*input_rows));
    vae_assembler assembler;
    memset(&assembler, 0, sizeof(assembler));
    int ok = input_rows != NULL;
    if (!ok) fail(error, error_size, "out of memory preparing video VAE tiles");
    ok = ok && ensure_pipeline_slots(decoder, error, error_size) &&
         assembler_init(&assembler, &decoder->y_axis, &decoder->x_axis,
                        chunks, sink, opaque, error, error_size);
    h3_gpu_tensor *outputs[2] = {vae->projected, decoder->slot_projected};
    h3_gpu_tensor *resident_latent = vae->latent;
    int chained = ok && gpu_op(vae, h3_gpu_begin(vae->gpu), error, error_size,
                               "begin pipelined video VAE decode");
    ok = chained;
    double input_seconds = 0.0, encode_seconds = 0.0, wait_seconds = 0.0;
    for (int job = 0; ok && job < jobs; job++) {
        int slot = job & 1;
        int tile = job % tile_count;
        double prepare_start = vae_seconds();
        fill_tile_input(decoder, latent, latent_time, job / tile_count,
                        tile / decoder->x_axis.count,
                        tile % decoder->x_axis.count, input_rows);
        ok = h3_gpu_tensor_write_f32(decoder->slot_latent[slot], input_rows,
                                     latent_elements);
        if (!ok) fail(error, error_size, "cannot stage video VAE tile input");
        double encode_start = vae_seconds();
        vae->latent = decoder->slot_latent[slot];
        vae->projected = outputs[slot];
        ok = ok && encode_resident_tile(vae, decoder->suffix_zero, error,
                                        error_size) &&
             gpu_op(vae, h3_gpu_continue(vae->gpu), error, error_size,
                    "commit video VAE tile");
        vae->latent = resident_latent;
        vae->projected = outputs[0];
        double encoded = vae_seconds();
        input_seconds += encode_start - prepare_start;
        encode_seconds += encoded - encode_start;
        if (!ok || !job) continue;
        ok = gpu_op(vae, h3_gpu_wait_oldest(vae->gpu), error, error_size,
                    "video VAE tile");
        wait_seconds += vae_seconds() - encoded;
        ok = ok && assembler_take(&assembler, job - 1,
                                  h3_gpu_tensor_contents(outputs[slot ^ 1]),
                                  error, error_size);
    }
    if (ok) {
        double wait_start = vae_seconds();
        ok = gpu_op(vae, h3_gpu_wait_oldest(vae->gpu), error, error_size,
                    "video VAE tile");
        wait_seconds += vae_seconds() - wait_start;
        ok = ok && assembler_take(&assembler, jobs - 1,
                                  h3_gpu_tensor_contents(
                                      outputs[(jobs - 1) & 1]),
                                  error, error_size);
    }
    /* Success leaves only the empty buffer h3_gpu_continue opened. */
    if (ok) ok = gpu_op(vae, h3_gpu_submit(vae->gpu), error, error_size,
                        "finish pipelined video VAE decode");
    else if (chained) drain_gpu_chain(vae->gpu);
    ok = ok && assembler_finish(&assembler, error, error_size);
    if (ok && assembler.profile) {
        h3_gpu_stats after = before;
        h3_gpu_get_stats(vae->gpu, &after);
        fprintf(stderr, "h3 profile: video VAE decode pipelined fused=%d "
                "tiles=%d chunks=%d wall=%.3fs gpu-wait=%.3fs root-gpu=%.3fs "
                "encode=%.3fs input=%.3fs unpack=%.3fs stitch=%.3fs "
                "blend=%.3fs sink=%.3fs\n", vae->fused, tile_count, chunks,
                vae_seconds() - started, wait_seconds,
                after.gpu_seconds - before.gpu_seconds, encode_seconds,
                input_seconds, assembler.unpack_seconds,
                assembler.stitch_seconds, assembler.blend_seconds,
                assembler.sink_seconds);
    }
    assembler_free(&assembler);
    free(input_rows);
    return ok;
}

static int valid_decode_arguments(const h3_video_vae_decoder *decoder,
                                  const float *normalized_latent,
                                  int latent_time) {
    return decoder && normalized_latent &&
           latent_time >= CHUNK_LATENT_TIME && !((latent_time - 2) % 5);
}

int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!output ||
        !valid_decode_arguments(decoder, normalized_latent, latent_time)) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    if (!h3_video_vae_pipelined())
        return decoder_decode_legacy(decoder, normalized_latent, latent_time,
                                     output, error, error_size);
    int output_frames = (latent_time - 2) / 5 * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    frame_copy copy = {NULL, (size_t)pixel_h * (size_t)pixel_w * 3};
    copy.rgb = malloc((size_t)output_frames * copy.frame_elements *
                      sizeof(*copy.rgb));
    if (!copy.rgb) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = decoder_decode_pipelined(decoder, normalized_latent, latent_time,
                                      copy_frames, &copy, error, error_size);
    if (ok) {
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = copy.rgb;
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    } else {
        free(copy.rgb);
    }
    if (!ok) h3_video_frames_free(output);
    return ok;
}

int h3_video_vae_decoder_decode_stream(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frame_sink sink, void *opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!sink ||
        !valid_decode_arguments(decoder, normalized_latent, latent_time)) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    if (h3_video_vae_pipelined())
        return decoder_decode_pipelined(decoder, normalized_latent,
                                        latent_time, sink, opaque,
                                        error, error_size);
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    int ok = decoder_decode_legacy(decoder, normalized_latent, latent_time,
                                   &frames, error, error_size);
    if (ok && !sink(frames.rgb, 0, frames.frames, frames.width,
                    frames.height, opaque)) {
        fail(error, error_size, "video frame consumer stopped the decode");
        ok = 0;
    }
    h3_video_frames_free(&frames);
    return ok;
}

void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder) {
    if (!decoder) return;
    free_tensor(&decoder->slot_latent[0]);
    free_tensor(&decoder->slot_latent[1]);
    free_tensor(&decoder->slot_projected);
    free_tensor(&decoder->suffix_zero);
    cleanup(&decoder->vae);
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    free(decoder);
}

static uint32_t selftest_hash(uint32_t index, uint32_t salt) {
    uint32_t state = index * 2654435761u + salt * 40503u + 12345u;
    state ^= state >> 13;
    state *= 0x5bd1e995u;
    return state ^ (state >> 15);
}

static void selftest_rows(int job, size_t count, float *rows) {
    for (size_t index = 0; index < count; index++)
        rows[index] = (float)((int)(selftest_hash((uint32_t)index,
            (uint32_t)job) % 24001u) - 12000) / 1000.0f;
}

int h3_video_vae_assembly_selftest(char *error, size_t error_size) {
    enum { CHUNKS = 3, TILE = 96, HEIGHT = 224, WIDTH = 160 };
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    if (!tile_axis_build(HEIGHT, TILE, &y_axis, error, error_size) ||
        !tile_axis_build(WIDTH, TILE, &x_axis, error, error_size)) {
        tile_axis_free(&y_axis);
        return 0;
    }
    int tile_count = y_axis.count * x_axis.count;
    int latent = TILE / SPATIAL_RATIO;
    size_t row_count = ((size_t)CHUNK_LATENT_TIME * (size_t)latent *
                        (size_t)latent + SUFFIX) * OUTPUT_PATCH;
    size_t frame_elements = (size_t)HEIGHT * WIDTH * 3;
    int output_frames = CHUNKS * 17 + 5;
    size_t output_elements = (size_t)output_frames * frame_elements;
    size_t tile_elements = (size_t)FIRST_CHUNK_FRAMES * TILE * TILE * 3;
    float *rows = malloc(row_count * sizeof(*rows));
    float *expected = malloc(output_elements * sizeof(*expected));
    float *overlap = malloc(5 * frame_elements * sizeof(*overlap));
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    frame_copy copy = {malloc(output_elements * sizeof(float)),
                       frame_elements};
    vae_assembler assembler;
    memset(&assembler, 0, sizeof(assembler));
    int ok = rows && expected && overlap && tiles && copy.rgb;
    for (int index = 0; ok && index < tile_count; index++)
        ok = (tiles[index] = malloc(tile_elements * sizeof(float))) != NULL;
    if (!ok) fail(error, error_size, "out of memory in VAE assembly test");
    /* Reference: the legacy order, one stitched chunk after another. */
    for (int chunk = 0; ok && chunk < CHUNKS; chunk++) {
        for (int tile = 0; tile < tile_count; tile++) {
            selftest_rows(chunk * tile_count + tile, row_count, rows);
            unpack_patch_rows(rows, latent, latent, FIRST_CHUNK_FRAMES, 0,
                              FIRST_CHUNK_FRAMES, tiles[tile]);
        }
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = stitch_tiles(tiles, &y_axis, &x_axis, FIRST_CHUNK_FRAMES,
                          &decoded, error, error_size);
        if (!ok) break;
        if (chunk) blend_temporal_overlap(decoded.rgb, overlap,
                                          frame_elements);
        memcpy(expected + (size_t)chunk * 17 * frame_elements, decoded.rgb,
               17 * frame_elements * sizeof(float));
        memcpy(overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(float));
        h3_video_frames_free(&decoded);
    }
    if (ok) memcpy(expected + (size_t)CHUNKS * 17 * frame_elements, overlap,
                   5 * frame_elements * sizeof(float));
    /* Pipelined host path: tiles stitched as they land, frames streamed. */
    ok = ok && assembler_init(&assembler, &y_axis, &x_axis, CHUNKS,
                              copy_frames, &copy, error, error_size);
    assembler.profile = 0;
    for (int job = 0; ok && job < CHUNKS * tile_count; job++) {
        selftest_rows(job, row_count, rows);
        ok = assembler_take(&assembler, job, rows, error, error_size);
    }
    ok = ok && assembler_finish(&assembler, error, error_size);
    if (ok && memcmp(expected, copy.rgb, output_elements * sizeof(float))) {
        fail(error, error_size, "pipelined VAE frames differ from legacy");
        ok = 0;
    }
    assembler_free(&assembler);
    for (int index = 0; tiles && index < tile_count; index++)
        free(tiles[index]);
    free(tiles);
    free(rows);
    free(expected);
    free(overlap);
    free(copy.rgb);
    tile_axis_free(&y_axis);
    tile_axis_free(&x_axis);
    return ok;
}

static int decode_chunked(const char *weight_directory,
                          const char *shader_source_path,
                          const float *normalized_latent, int latent_time,
                          int latent_height, int latent_width,
                          const float *latent_mean, const float *latent_std,
                          int tile_pixels,
                          h3_video_vae_progress progress, void *progress_opaque,
                          h3_video_frames *output, char *error,
                          size_t error_size) {
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int ok = tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                             &y_axis,
                             error, error_size) &&
             tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                             &x_axis,
                             error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3: video VAE tiles %dx%d at %d pixels\n",
                x_axis.count, y_axis.count, tile_pixels);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = y_axis.length / SPATIAL_RATIO;
    vae.latent_w = x_axis.length / SPATIAL_RATIO;
    vae.latent_t = CHUNK_LATENT_TIME;
    vae.output_frames = FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = open_weights(&vae, weight_directory, error, error_size);
    if (vae.weights)
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu) vae.precision = configured_precision(vae.gpu);
    if (vae.gpu) vae.bf16_attention = vae.precision != VAE_F32 &&
        !getenv("H3_VAE_F32_ATTENTION");
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    ok = vae.weights && vae.gpu &&
         load_resident_weights(&vae, progress, progress_opaque,
                               error, error_size) &&
         prepare_rope(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size);
    int tile_count = y_axis.count * x_axis.count;
    int chunks = (latent_time - 2) / 5;
    int output_frames = chunks * 17 + 5;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = ok ? malloc(output_elements * sizeof(*final_rgb)) : NULL;
    float *temporal_overlap = ok ? malloc(5 * frame_elements *
                                          sizeof(*temporal_overlap)) : NULL;
    if (ok && (!final_rgb || !temporal_overlap)) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        ok = 0;
    }
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
        if (!tiles) {
            fail(error, error_size, "out of memory retaining video VAE tiles");
            ok = 0;
            break;
        }
        for (int tile_y = 0; tile_y < y_axis.count && ok; tile_y++)
            for (int tile_x = 0; tile_x < x_axis.count && ok; tile_x++) {
                float *input = extract_latent_tile(normalized_latent,
                    latent_time, latent_height, latent_width, chunk * 5,
                    y_axis.starts[tile_y] / SPATIAL_RATIO,
                    x_axis.starts[tile_x] / SPATIAL_RATIO,
                    CHUNK_LATENT_TIME, vae.latent_h, vae.latent_w,
                    error, error_size);
                if (!input) {
                    ok = 0;
                    break;
                }
                free_tensor(&vae.latent);
                ok = prepare_input(&vae, input, latent_mean, latent_std,
                                   error, error_size) &&
                     run_resident_tile(&vae, error, error_size);
                free(input);
                if (!ok) break;
                h3_video_frames tile;
                memset(&tile, 0, sizeof(tile));
                ok = unpack_frames(&vae, &tile, error, error_size);
                if (ok) {
                    int index = tile_y * x_axis.count + tile_x;
                    tiles[index] = tile.rgb;
                }
            }
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        if (ok) ok = stitch_tiles(tiles, &y_axis, &x_axis,
                                  FIRST_CHUNK_FRAMES, &decoded,
                                  error, error_size);
        for (int index = 0; index < tile_count; index++) free(tiles[index]);
        free(tiles);
        if (!ok) {
            h3_video_frames_free(&decoded);
            break;
        }
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    cleanup(&vae);
    tile_axis_free(&y_axis); tile_axis_free(&x_axis);
    return ok;
}

void h3_video_frames_free(h3_video_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_time < 2 ||
        (latent_time != 2 && (latent_time < CHUNK_LATENT_TIME ||
                              (latent_time - 2) % 5)) ||
        latent_height < 1 || latent_width < 1 ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX ||
        (uint32_t)(latent_height * latent_width) >
            (UINT32_MAX - SUFFIX) / CHUNK_LATENT_TIME) {
        fail(error, error_size, "invalid first-chunk video VAE arguments");
        return 0;
    }
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.latent_t = latent_time;
    vae.output_frames = latent_time == 2 ? 5 : FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * latent_height * latent_width);
    vae.sequence = vae.patches + SUFFIX;
    float latent_mean[LATENT_CHANNELS], latent_std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, latent_mean, latent_std,
                                   error, error_size)) return 0;
    int tile_pixels = configured_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO);
    if (latent_time == 2 &&
        (latent_height > TILE_PIXELS / SPATIAL_RATIO ||
         latent_width > TILE_PIXELS / SPATIAL_RATIO)) {
        fail(error, error_size,
             "the two-token diagnostic VAE path supports one spatial tile");
        return 0;
    }
    if (latent_time > CHUNK_LATENT_TIME ||
        latent_height > tile_pixels / SPATIAL_RATIO ||
        latent_width > tile_pixels / SPATIAL_RATIO) {
        int ok = decode_chunked(weight_directory, shader_source_path,
                                normalized_latent, latent_time, latent_height,
                                latent_width, latent_mean, latent_std,
                                tile_pixels, progress,
                                progress_opaque, output, error, error_size);
        if (!ok) h3_video_frames_free(output);
        return ok;
    }
    vae.weights = open_weights(&vae, weight_directory, error, error_size);
    if (!vae.weights) return 0;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu) vae.precision = configured_precision(vae.gpu);
    if (vae.gpu) vae.bf16_attention = vae.precision != VAE_F32 &&
        !getenv("H3_VAE_F32_ATTENTION");
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    int ok = vae.gpu &&
        load_input_weights(&vae, error, error_size) &&
        prepare_input(&vae, normalized_latent, latent_mean, latent_std,
                      error, error_size) &&
        prepare_rope(&vae, error, error_size) &&
        allocate_activations(&vae, error, error_size) &&
        run_decoder(&vae, progress, progress_opaque, error, error_size) &&
        unpack_frames(&vae, output, error, error_size);
    if (!ok) h3_video_frames_free(output);
    cleanup(&vae);
    return ok;
}
