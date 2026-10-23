/* Flat wrapper over h3.h; see h3_flat.h. */
#include "h3_flat.h"
#include "h3_mutex.h"

#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { FLAT_OUTPUT, FLAT_FIRST, FLAT_LAST, FLAT_MOTION, FLAT_SAVE,
       FLAT_LORA, FLAT_STRINGS };

struct h3_flat_params {
    h3_params params;
    char *strings[FLAT_STRINGS];
    h3_reference *references;
    char **reference_strings;
    size_t reference_count;
    h3_flat_frame_fn on_frame;
    h3_progress_callback on_progress;
    void *opaque;
};

typedef struct {
    const char *name;
    size_t offset;
} flat_field;

#define FIELD(name) { #name, offsetof(h3_params, name) }
static const flat_field int_fields[] = {
    FIELD(width), FIELD(height), FIELD(frames), FIELD(steps),
    FIELD(denoise_reuse), FIELD(dit_layers), FIELD(core_reuse),
    FIELD(token_reduction), FIELD(use_int8_row_fc2),
    FIELD(use_reference_rope), FIELD(ssd_streaming), FIELD(render_width),
    FIELD(render_height), FIELD(use_slower_bf16_mlp),
    FIELD(use_slower_bf16_qkv), FIELD(use_slower_bf16_attention_output),
    FIELD(use_slower_row_major_attention_output),
    FIELD(use_slower_unfused_int8_inputs), FIELD(use_slower_unfused_qkv_rope),
    FIELD(use_slower_scalar_qkv_rms), FIELD(use_slower_uncached_int8_scales),
    FIELD(use_slower_dynamic_fc1_k), FIELD(use_slower_grouped_quantizer),
    FIELD(preview_denoise), FIELD(sampler), FIELD(scheduler),
};
#undef FIELD

const char *h3_flat_version(void) { return H3_VERSION; }

h3_flat_params *h3_flat_params_new(void) {
    h3_flat_params *flat = calloc(1, sizeof(*flat));
    if (!flat) return NULL;
    h3_params defaults = H3_PARAMS_DEFAULT;
    flat->params = defaults;
    return flat;
}

void h3_flat_params_clear_references(h3_flat_params *flat) {
    if (!flat) return;
    for (size_t i = 0; i < flat->reference_count * 2; i++)
        free(flat->reference_strings[i]);
    free(flat->reference_strings);
    free(flat->references);
    flat->reference_strings = NULL;
    flat->references = NULL;
    flat->reference_count = 0;
}

void h3_flat_params_free(h3_flat_params *flat) {
    if (!flat) return;
    for (int i = 0; i < FLAT_STRINGS; i++) free(flat->strings[i]);
    h3_flat_params_clear_references(flat);
    free(flat);
}

int h3_flat_params_set_int(h3_flat_params *flat, const char *name, int value) {
    if (!flat || !name) return 0;
    if (!strcmp(name, "reference_image_size")) {
        flat->params.reference_image_size = (h3_reference_image_size)value;
        return 1;
    }
    for (size_t i = 0; i < sizeof(int_fields) / sizeof(int_fields[0]); i++) {
        if (!strcmp(name, int_fields[i].name)) {
            *(int *)((char *)&flat->params + int_fields[i].offset) = value;
            return 1;
        }
    }
    return 0;
}

int h3_flat_params_set_u64(h3_flat_params *flat, const char *name,
                           uint64_t value) {
    if (!flat || !name || strcmp(name, "seed")) return 0;
    flat->params.seed = value;
    return 1;
}

int h3_flat_params_set_double(h3_flat_params *flat, const char *name,
                              double value) {
    if (!flat || !name) return 0;
    if (!strcmp(name, "video_shift")) flat->params.video_shift = value;
    else if (!strcmp(name, "lora_scale")) flat->params.lora_scale = value;
    else return 0;
    return 1;
}

static char *copy_or_null(const char *value) {
    return value && *value ? strdup(value) : NULL;
}

int h3_flat_params_set_string(h3_flat_params *flat, const char *name,
                              const char *value) {
    if (!flat || !name) return 0;
    static const char *const names[FLAT_STRINGS] = {
        "output_path", "first_frame", "last_frame", "motion_context",
        "save_context", "lora_path"
    };
    for (int i = 0; i < FLAT_STRINGS; i++) {
        if (strcmp(name, names[i])) continue;
        char *copy = copy_or_null(value);
        if (value && *value && !copy) return 0;
        free(flat->strings[i]);
        flat->strings[i] = copy;
        return 1;
    }
    return 0;
}

int h3_flat_params_add_reference(h3_flat_params *flat, int kind,
                                 const char *path, const char *audio_path,
                                 int include_embedded_audio) {
    if (!flat || !path || !*path) return 0;
    size_t count = flat->reference_count + 1;
    h3_reference *references = realloc(flat->references,
                                       count * sizeof(*references));
    if (!references) return 0;
    flat->references = references;
    char **strings = realloc(flat->reference_strings,
                             count * 2 * sizeof(*strings));
    if (!strings) return 0;
    flat->reference_strings = strings;
    char *path_copy = strdup(path);
    char *audio_copy = copy_or_null(audio_path);
    if (!path_copy || (audio_path && *audio_path && !audio_copy)) {
        free(path_copy);
        free(audio_copy);
        return 0;
    }
    strings[2 * (count - 1)] = path_copy;
    strings[2 * (count - 1) + 1] = audio_copy;
    h3_reference reference = {(h3_reference_kind)kind, path_copy, audio_copy,
                              include_embedded_audio};
    references[count - 1] = reference;
    flat->reference_count = count;
    return 1;
}

void h3_flat_params_set_callbacks(h3_flat_params *flat,
                                  h3_flat_frame_fn on_frame,
                                  h3_progress_callback on_progress,
                                  void *opaque) {
    if (!flat) return;
    flat->on_frame = on_frame;
    flat->on_progress = on_progress;
    flat->opaque = opaque;
}

static int flat_frame_trampoline(const h3_frame *frame, void *opaque) {
    const h3_flat_params *flat = opaque;
    return flat->on_frame(frame->width, frame->height, frame->stride,
                          frame->rgb, frame->frame_index, frame->frame_count,
                          frame->denoise_step, frame->denoise_steps,
                          flat->opaque);
}

static int flat_progress_trampoline(const char *phase, int completed,
                                    int total, void *opaque) {
    const h3_flat_params *flat = opaque;
    return flat->on_progress(phase, completed, total, flat->opaque);
}

h3_result *h3_flat_generate(h3_ctx *ctx, const char *prompt,
                            const h3_flat_params *flat,
                            const char *shader_dir) {
    if (!ctx || !flat) return NULL;
    h3_params params = flat->params;
    params.output_path = flat->strings[FLAT_OUTPUT];
    params.first_frame = flat->strings[FLAT_FIRST];
    params.last_frame = flat->strings[FLAT_LAST];
    params.motion_context = flat->strings[FLAT_MOTION];
    params.save_context = flat->strings[FLAT_SAVE];
    params.lora_path = flat->strings[FLAT_LORA];
    params.references = flat->references;
    params.reference_count = flat->reference_count;
    params.on_frame = flat->on_frame ? flat_frame_trampoline : NULL;
    params.on_progress = flat->on_progress ? flat_progress_trampoline : NULL;
    params.callback_opaque = (void *)flat;

    int saved_cwd = -1;
    if (shader_dir && *shader_dir) {
        saved_cwd = open(".", O_RDONLY | O_CLOEXEC);
        if (saved_cwd < 0 || chdir(shader_dir) != 0) {
            if (saved_cwd >= 0) close(saved_cwd);
            return NULL;
        }
    }
    h3_result *result = h3_generate(ctx, prompt, &params);
    if (saved_cwd >= 0) {
        if (fchdir(saved_cwd) != 0) { /* nothing better to do */ }
        close(saved_cwd);
    }
    return result;
}

int64_t h3_flat_result_get(const h3_result *result, const char *name) {
    if (!result || !name) return -1;
    if (!strcmp(name, "width")) return result->width;
    if (!strcmp(name, "height")) return result->height;
    if (!strcmp(name, "frames")) return result->frames;
    if (!strcmp(name, "fps")) return result->fps;
    if (!strcmp(name, "sample_rate")) return result->sample_rate;
    if (!strcmp(name, "seed")) return (int64_t)result->seed;
    return -1;
}

void *h3_flat_lock_acquire(void) {
    h3_mutex *mutex = malloc(sizeof(*mutex));
    if (!mutex) return NULL;
    if (!h3_mutex_acquire(mutex)) {
        free(mutex);
        return NULL;
    }
    return mutex;
}

void h3_flat_lock_release(void *lock) {
    if (!lock) return;
    h3_mutex_release(lock);
    free(lock);
}
