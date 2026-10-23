/* Flat, struct-free wrapper over h3.h for foreign-function callers (ctypes,
 * cffi, Swift). Every h3_params field is set by name, so a binding never
 * mirrors the positional struct and survives fields being appended to it.
 * Strings handed to a setter are copied; the caller may free them at once. */
#ifndef H3_FLAT_H
#define H3_FLAT_H

#include <stdint.h>
#include "h3.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_flat_params h3_flat_params;

/* Frame delivery without the h3_frame struct. denoise_step is -1 for a final
 * decoded frame and the step index for a denoising preview. Return nonzero to
 * cancel the generation. */
typedef int (*h3_flat_frame_fn)(int width, int height, int stride,
                                const uint8_t *rgb, int frame_index,
                                int frame_count, int denoise_step,
                                int denoise_steps, void *opaque);

const char *h3_flat_version(void);

/* A params block holding H3_PARAMS_DEFAULT. */
h3_flat_params *h3_flat_params_new(void);
void h3_flat_params_free(h3_flat_params *params);

/* Each setter returns 1, or 0 for an unknown name (a typo is never silent).
 * Integer names: width height frames steps denoise_reuse dit_layers
 * core_reuse token_reduction use_int8_row_fc2 use_reference_rope
 * ssd_streaming render_width render_height preview_denoise sampler scheduler
 * reference_image_size (and the use_slower_* switches).
 * u64: seed.  double: video_shift lora_scale.
 * string (NULL or "" clears): output_path first_frame last_frame
 * motion_context save_context lora_path. */
int h3_flat_params_set_int(h3_flat_params *params, const char *name, int value);
int h3_flat_params_set_u64(h3_flat_params *params, const char *name,
                           uint64_t value);
int h3_flat_params_set_double(h3_flat_params *params, const char *name,
                              double value);
int h3_flat_params_set_string(h3_flat_params *params, const char *name,
                              const char *value);

/* Append one ordered reference (kind is an h3_reference_kind value). */
int h3_flat_params_add_reference(h3_flat_params *params, int kind,
                                 const char *path, const char *audio_path,
                                 int include_embedded_audio);
void h3_flat_params_clear_references(h3_flat_params *params);

void h3_flat_params_set_callbacks(h3_flat_params *params,
                                  h3_flat_frame_fn on_frame,
                                  h3_progress_callback on_progress,
                                  void *opaque);

/* h3_generate run with the working directory set to shader_dir (the engine
 * reads h3_shaders.metal relative to it) and restored afterwards. All paths in
 * params and model_dir should be absolute. NULL shader_dir leaves cwd alone. */
h3_result *h3_flat_generate(h3_ctx *ctx, const char *prompt,
                            const h3_flat_params *params,
                            const char *shader_dir);

/* Result fields by name: width height frames fps sample_rate seed. */
int64_t h3_flat_result_get(const h3_result *result, const char *name);

/* The CLI's process-wide flock ($TMPDIR/h3.c.lock): blocks until no other h3
 * process runs. Returns a handle or NULL. */
void *h3_flat_lock_acquire(void);
void h3_flat_lock_release(void *lock);

#ifdef __cplusplus
}
#endif
#endif
