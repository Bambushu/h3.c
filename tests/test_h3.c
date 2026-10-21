#include "h3_host.h"
#include "h3_dit.h"
#include "h3_dit_cache.h"
#include "h3_metal.h"
#include "h3_safetensors.h"
#include "h3_terminal.h"
#include "h3_ffmpeg.h"
#include "h3_video_encoder.h"
#include "h3_video_vae.h"
#include "h3_motion_context.h"

#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>

static int tests_run;

#define CHECK(condition) do { \
    tests_run++; \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

static int close_enough(double value, double expected, double tolerance) {
    return fabs(value - expected) <= tolerance;
}

static void test_temporal_and_canvas(void) {
    const struct { int requested, frames, video_t, audio_t; } cases[] = {
        {5, 5, 2, 8}, {6, 22, 7, 37}, {22, 22, 7, 37},
        {23, 39, 12, 65}, {56, 56, 17, 93}, {124, 124, 37, 207},
        {361, 362, 107, 603}, {362, 362, 107, 603}
    };
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        h3_temporal_shape got = h3_temporal(cases[index].requested);
        CHECK(got.frame_count == cases[index].frames);
        CHECK(got.video_t == cases[index].video_t);
        CHECK(got.audio_t == cases[index].audio_t);
    }
    CHECK(h3_align_frame_count(INT32_MAX) == 0);
    CHECK(h3_video_encoder_latent_t(1) == 1);
    CHECK(h3_video_encoder_latent_t(5) == 2);
    CHECK(h3_video_encoder_latent_t(22) == 6);
    CHECK(h3_video_encoder_latent_t(39) == 10);
    int width, height;
    CHECK(h3_adapt_canvas(1920, 1080, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_adapt_canvas(1080, 1920, &width, &height));
    CHECK(width == 768 && height == 1344);
    CHECK(h3_adapt_canvas(864, 480, &width, &height));
    CHECK(width == 1376 && height == 768);
    CHECK(h3_adapt_canvas(32, 32, &width, &height));
    CHECK(width == 768 && height == 768);
    CHECK(h3_reference_image_canvas(1920, 1080, 512, 512, 0,
                                    &width, &height));
    CHECK(width == 672 && height == 384);
    CHECK(h3_reference_image_canvas(640, 480, 1024, 1024, 0,
                                    &width, &height));
    CHECK(width == 640 && height == 480);
    CHECK(h3_reference_image_canvas(4096, 2160, 512, 512, 2048,
                                    &width, &height));
    CHECK(width == 3872 && height == 2048);
    CHECK(!h3_reference_image_canvas(0, 480, 512, 512, 0,
                                     &width, &height));
    CHECK(h3_reference_video_canvas(1920, 1080, &width, &height));
    CHECK(width == 1344 && height == 768);
    CHECK(h3_reference_video_canvas(640, 360, &width, &height));
    CHECK(width == 640 && height == 352);
    CHECK(!h3_reference_video_canvas(0, 360, &width, &height));
}

static void test_schedule(void) {
    h3_params defaults = H3_PARAMS_DEFAULT;
    CHECK(defaults.steps == 20);
    CHECK(defaults.use_reference_rope == 0);
    CHECK(defaults.sampler == H3_SAMPLER_EULER);
    CHECK(defaults.scheduler == H3_SCHEDULER_LINEAR);

    h3_sigma_schedule schedule;
    CHECK(h3_schedule_build(20, &schedule));
    CHECK(schedule.steps == 20);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[20] == 0.0f && schedule.audio[20] == 0.0f);
    CHECK(close_enough(schedule.video[1], 0.995633185, 1e-7));
    CHECK(close_enough(schedule.audio[1], 0.982758582, 1e-7));
    CHECK(close_enough(schedule.video[19], 0.387096792, 1e-7));
    CHECK(close_enough(schedule.audio[19], 0.136363640, 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }

    /* Selectable base schedules. simple is the released linear grid when the
     * step count divides the 1000-point ComfyUI grid; beta and beta57 warp the
     * same grid and must still be monotone and terminate at zero. */
    h3_sigma_schedule linear, simple, beta, beta57;
    CHECK(h3_serving_schedule_build_ex(8, H3_SCHEDULER_LINEAR, &linear));
    CHECK(h3_serving_schedule_build_ex(8, H3_SCHEDULER_SIMPLE, &simple));
    CHECK(h3_serving_schedule_build_ex(8, H3_SCHEDULER_BETA, &beta));
    CHECK(h3_serving_schedule_build_ex(8, H3_SCHEDULER_BETA57, &beta57));
    for (int index = 0; index <= 8; index++) {
        CHECK(simple.video[index] == linear.video[index]);
        CHECK(simple.audio[index] == linear.audio[index]);
        CHECK(beta.video[index] >= beta57.video[index]);
    }
    for (int index = 0; index < 8; index++) {
        CHECK(beta.video[index] > beta.video[index + 1]);
        CHECK(beta57.video[index] > beta57.video[index + 1]);
        CHECK(beta.audio[index] > beta.audio[index + 1]);
        CHECK(beta57.audio[index] > beta57.audio[index + 1]);
    }
    CHECK(beta.video[8] == 0.0f && beta57.video[8] == 0.0f);
    CHECK(beta.video[0] == 1.0f && beta57.video[0] == 1.0f);
    /* Pinned against ComfyUI beta_scheduler(0.6,0.6) and the RES4LYF beta57
     * registration beta_scheduler(0.5,0.7) at four steps. */
    h3_sigma_schedule beta4, beta57_4;
    CHECK(h3_serving_schedule_build_ex(4, H3_SCHEDULER_BETA, &beta4));
    CHECK(h3_serving_schedule_build_ex(4, H3_SCHEDULER_BETA57, &beta57_4));
    CHECK(close_enough(beta4.video[1], 0.9825119, 1e-6));
    CHECK(close_enough(beta4.video[2], 0.9233605, 1e-6));
    CHECK(close_enough(beta4.video[3], 0.7207329, 1e-6));
    CHECK(close_enough(beta57_4.video[1], 0.9686099, 1e-6));
    CHECK(close_enough(beta57_4.video[2], 0.8719390, 1e-6));
    CHECK(close_enough(beta57_4.video[3], 0.5631350, 1e-6));
    CHECK(!h3_serving_schedule_build_ex(4, 99, &beta4));
    /* The video sigma shift is a parameter: 12 reproduces the released grid
     * byte for byte, another shift moves the video grid and leaves audio. */
    h3_sigma_schedule shift12, shift6, beta_shift6;
    CHECK(h3_serving_schedule_build_shifted(8, H3_SCHEDULER_LINEAR, 12.0,
                                            &shift12));
    CHECK(h3_serving_schedule_build_shifted(8, H3_SCHEDULER_LINEAR, 6.0,
                                            &shift6));
    CHECK(h3_serving_schedule_build_shifted(8, H3_SCHEDULER_BETA57, 6.0,
                                            &beta_shift6));
    for (int index = 0; index <= 8; index++) {
        CHECK(shift12.video[index] == linear.video[index]);
        CHECK(shift12.audio[index] == linear.audio[index]);
        CHECK(shift6.audio[index] == linear.audio[index]);
    }
    for (int index = 1; index < 8; index++) {
        CHECK(shift6.video[index] < linear.video[index]);
        CHECK(beta_shift6.video[index] < beta57.video[index]);
    }
    CHECK(shift6.video[0] == 1.0f && shift6.video[8] == 0.0f);
    CHECK(beta_shift6.video[0] == 1.0f && beta_shift6.video[8] == 0.0f);
    CHECK(!h3_serving_schedule_build_shifted(8, H3_SCHEDULER_LINEAR, 0.5,
                                             &shift6));
    CHECK(!h3_serving_schedule_build_shifted(8, H3_SCHEDULER_LINEAR, 31.0,
                                             &shift6));
    double shifted = h3_time_shift_sigma(0.5, 12.0, 3.0);
    double back = h3_time_shift_sigma(shifted, 3.0, 12.0);
    CHECK(close_enough(back, 0.5, 1e-12));
    CHECK(h3_time_shift_slope(0.5, 12.0, 3.0) > 0.0);

    CHECK(h3_serving_schedule_build(50, &schedule));
    CHECK(schedule.steps == 50);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(schedule.video[50] == 0.0f && schedule.audio[50] == 0.0f);
    float base = 0.5f;
    CHECK(close_enough(schedule.video[25],
        12.0f * base / (1.0f + 11.0f * base), 1e-7));
    CHECK(close_enough(schedule.audio[25],
        3.0f * base / (1.0f + 2.0f * base), 1e-7));
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }

    CHECK(h3_serving_schedule_build(4, &schedule));
    CHECK(schedule.steps == 4);
    CHECK(schedule.video[0] == 1.0f && schedule.audio[0] == 1.0f);
    CHECK(close_enough(schedule.video[1], 36.0 / 37.0, 1e-7));
    CHECK(close_enough(schedule.audio[1], 9.0 / 10.0, 1e-7));
    CHECK(schedule.video[4] == 0.0f && schedule.audio[4] == 0.0f);
    for (int index = 0; index < schedule.steps; index++) {
        CHECK(schedule.video[index] > schedule.video[index + 1]);
        CHECK(schedule.audio[index] > schedule.audio[index + 1]);
    }
    CHECK(!h3_serving_schedule_build(1, &schedule));
    CHECK(!h3_serving_schedule_build(H3_MAX_STEPS + 1, &schedule));
}

static void test_dit_reuse_schedule(void) {
    uint8_t selected[50];
    const int aggressive[] = {0, 3, 6, 9, 12, 15, 18, 19};
    CHECK(h3_dit_reuse_schedule(20, 3, selected, sizeof(selected)) == 8);
    for (int step = 0; step < 20; step++) {
        int expected = 0;
        for (size_t index = 0;
             index < sizeof(aggressive) / sizeof(*aggressive); index++)
            expected |= step == aggressive[index];
        CHECK(selected[step] == expected);
    }
    CHECK(h3_dit_reuse_schedule(20, 2, selected, sizeof(selected)) == 11);
    for (int step = 0; step < 20; step++)
        CHECK(selected[step] == (step % 2 == 0 || step == 19));
    CHECK(h3_dit_reuse_schedule(50, 3, selected, sizeof(selected)) == 18);
    for (int step = 0; step < 50; step++)
        CHECK(selected[step] == (step % 3 == 0 || step == 49));
    CHECK(h3_dit_reuse_schedule(20, 1, selected, sizeof(selected)) == 20);
    CHECK(h3_dit_reuse_schedule(20, 3, selected, 19) == -1);
}

static void check_segments(const h3_layout *layout,
                           const size_t (*bounds)[2],
                           const h3_segment_kind *kinds, size_t count) {
    CHECK(layout->segment_count == count);
    for (size_t index = 0; index < count; index++) {
        CHECK(layout->segments[index].start == bounds[index][0]);
        CHECK(layout->segments[index].stop == bounds[index][1]);
        CHECK(layout->segments[index].kind == kinds[index]);
        if (index) CHECK(layout->segments[index - 1].stop == bounds[index][0]);
    }
}

static void position_checksums(const h3_layout *layout, double sums[3],
                               double weighted[3]) {
    memset(sums, 0, 3 * sizeof(*sums));
    memset(weighted, 0, 3 * sizeof(*weighted));
    for (size_t index = 0; index < layout->seq_len; index++) {
        const double values[3] = {layout->positions[index].t,
                                  layout->positions[index].h,
                                  layout->positions[index].w};
        for (int axis = 0; axis < 3; axis++) {
            sums[axis] += values[axis];
            weighted[axis] += (double)(index + 1) * values[axis];
        }
    }
}

static void test_layout_tiny(void) {
    h3_layout_spec spec = {12, 2, 2, 2, 8, 5, NULL, 0, NULL, 0, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 30);
    const size_t bounds[][2] = {{0, 12}, {12, 28}, {28, 30}};
    const h3_segment_kind kinds[] = {H3_SEG_TEXT, H3_SEG_AUDIO, H3_SEG_VIDEO};
    check_segments(&layout, bounds, kinds, 3);
    CHECK(layout.img_target_rows == 2 && layout.audio_target_rows == 16);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 339.6666666666667, 1e-10));
    CHECK(close_enough(weighted[0], 6498.0, 1e-10));
    CHECK(sums[1] == 0.0 && sums[2] == 0.0);
    h3_layout_free(&layout);
}

static void test_layout_fl2va(void) {
    int keyframes[] = {0, 55};
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56,
                           keyframes, 2, NULL, 0, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8009);
    const size_t bounds[][2] = {
        {0, 128}, {128, 533}, {533, 938}, {938, 1124}, {1124, 8009}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 5);
    CHECK(layout.img_cond_rows == 810);
    CHECK(layout.img_target_rows == 6885);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 1360927.0, 1e-6));
    CHECK(close_enough(weighted[0], 5883121758.0, 1e-4));
    CHECK(close_enough(sums[1], 117002.11801356057, 1e-7));
    CHECK(close_enough(sums[2], 119830.2393846486, 1e-7));
    h3_layout_free(&layout);
}

static void test_layout_motion_context(void) {
    /* 22 pinned frames = 7 latent steps; 37 audio steps ending at the join. */
    h3_layout_spec spec = {128, 17, 30, 54, 93, 56, NULL, 0, NULL, 0, 7, 37, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 10108);
    const size_t bounds[][2] = {
        {0, 128}, {128, 2963}, {2963, 3037}, {3037, 3223}, {3223, 10108}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_REF_AUDIO, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 5);
    CHECK(layout.img_cond_rows == 2835 && layout.audio_cond_rows == 74);
    CHECK(layout.img_target_rows == 6885 && layout.audio_target_rows == 186);
    /* Every pinned row shares its (t, h, w) with the target row it stands
     * in for: the head occupies the target's own first seven steps. */
    int same = 1;
    for (size_t row = 0; row < 2835; row++) {
        const h3_position *pinned = &layout.positions[128 + row];
        const h3_position *target = &layout.positions[3223 + row];
        if (pinned->t != target->t || pinned->h != target->h ||
            pinned->w != target->w) same = 0;
    }
    CHECK(same);
    CHECK(layout.positions[128].t == 128.0);
    CHECK(close_enough(layout.positions[128 + 6 * 405].t, 158.0, 1e-12));
    /* The audio window: 37 steps from the origin, stereo rows on the target
     * audio w positions, identical to the first 37 target audio steps. */
    same = 1;
    for (size_t step = 0; step < 37; step++) {
        for (int stereo = 0; stereo < 2; stereo++) {
            const h3_position *pinned =
                &layout.positions[2963 + (size_t)stereo * 37 + step];
            const h3_position *target =
                &layout.positions[3037 + (size_t)stereo * 93 + step];
            if (pinned->t != target->t || pinned->w != target->w) same = 0;
        }
    }
    CHECK(same);
    h3_layout_free(&layout);
    /* A negative offset (a previous clip whose audio grid stopped short of
     * its last frame) starts one step before the origin. */
    spec.head_audio_offset = -1.0;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.positions[2963].t == 127.0);
    CHECK(layout.positions[2963 + 36].t == 163.0);
    h3_layout_free(&layout);
    /* The pinned run must leave room to generate, and it replaces the
     * first-frame keyframe rather than stacking on it. */
    spec.head_audio_offset = 0.0;
    spec.head_steps = 17;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    spec.head_steps = 7;
    int first = 0;
    spec.keyframes = &first;
    spec.keyframe_count = 1;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    int last = 55;
    spec.keyframes = &last;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.img_cond_rows == 2835 + 405);
    h3_layout_free(&layout);
}

static void test_motion_context_file(void) {
    CHECK(h3_video_head_steps(22) == 7 && h3_video_head_steps(5) == 2 &&
          h3_video_head_steps(39) == 12 && h3_video_head_steps(56) == 17 &&
          h3_video_head_steps(1) == 1 && h3_video_head_steps(4) == 0 &&
          h3_video_head_steps(0) == 0);
    /* A 39-frame clip: 12 latent steps on a 4x6 canvas, 65 audio steps. */
    const int video_t = 12, lh = 4, lw = 6, audio_t = 65, frame_count = 39;
    size_t video_count = (size_t)24 * video_t * lh * lw;
    size_t audio_count = (size_t)32 * 2 * audio_t;
    float *video = malloc(video_count * sizeof(*video));
    float *audio = malloc(audio_count * sizeof(*audio));
    for (size_t index = 0; index < video_count; index++)
        video[index] = (float)(index % 977) * 0.01f;
    for (size_t index = 0; index < audio_count; index++)
        audio[index] = (float)(index % 311) * -0.02f;
    char path[] = "/tmp/h3-motion-context-test.h3ctx";
    char error[256];
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, audio_t,
                                 frame_count, error, sizeof(error)));
    h3_motion_context context;
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.from_latents && context.frames == 22 && context.steps == 7);
    CHECK(context.audio_t == 37 && context.audio_offset == 0.0);
    CHECK(h3_motion_context_encode(&context, path, lw * 16, lh * 16, NULL,
                                   NULL, NULL, NULL, NULL, NULL, NULL,
                                   error, sizeof(error)));
    /* The rows are the patchified last 7 steps and the packed last 37 audio
     * steps, element for element. */
    size_t plane = (size_t)lh * lw;
    float *tail = malloc((size_t)24 * 7 * plane * sizeof(*tail));
    for (int channel = 0; channel < 24; channel++)
        memcpy(tail + (size_t)channel * 7 * plane,
               video + ((size_t)channel * video_t + 5) * plane,
               7 * plane * sizeof(*tail));
    size_t row_elements = 7 * plane / 4 * 96;
    float *rows = malloc(row_elements * sizeof(*rows));
    CHECK(h3_dit_patchify_video(tail, 24, 7, lh, lw, rows, row_elements));
    CHECK(context.video_elements == row_elements &&
          memcmp(context.video_rows, rows, row_elements * sizeof(*rows)) == 0);
    float *audio_tail = malloc((size_t)32 * 2 * 37 * sizeof(*audio_tail));
    for (int channel = 0; channel < 32; channel++)
        for (int stereo = 0; stereo < 2; stereo++)
            memcpy(audio_tail + ((size_t)channel * 2 + stereo) * 37,
                   audio + ((size_t)channel * 2 + stereo) * audio_t + 28,
                   37 * sizeof(*audio_tail));
    float *audio_rows = malloc((size_t)32 * 2 * 37 * sizeof(*audio_rows));
    CHECK(h3_dit_pack_audio(audio_tail, 32, 37, audio_rows, (size_t)32 * 2 * 37));
    CHECK(context.audio_elements == (size_t)32 * 2 * 37 &&
          memcmp(context.audio_rows, audio_rows,
                 context.audio_elements * sizeof(*audio_rows)) == 0);
    h3_motion_context_free(&context);
    /* A 107-frame clip allocates 178 audio steps for 178.33 frames' worth:
     * the window ends a third of a step early, so it starts one step before
     * the origin (round(36.33) - 37). */
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, 178,
                                 107, error, sizeof(error)));
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.audio_offset == -1.0);
    /* A 124-frame clip overshoots by a third: round(37.0) - 37 = 0. */
    CHECK(h3_motion_context_save(path, video, video_t, lh, lw, audio, 207,
                                 124, error, sizeof(error)));
    CHECK(h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    CHECK(context.audio_offset == 0.0);
    /* Another canvas is refused; a tail off the cycle is refused. */
    CHECK(!h3_motion_context_plan(path, lh, lw + 2, &context, error, sizeof(error)));
    CHECK(h3_motion_context_save(path, video, 11, lh, lw, audio, audio_t,
                                 frame_count, error, sizeof(error)));
    CHECK(!h3_motion_context_plan(path, lh, lw, &context, error, sizeof(error)));
    remove(path);
    /* A video file plans a 37-step window from the origin without reading it. */
    CHECK(!h3_motion_context_plan("/nonexistent/clip.mp4", lh, lw, &context,
                                  error, sizeof(error)));
    FILE *stub = fopen("/tmp/h3-motion-context-test.mp4", "wb");
    CHECK(stub && fputs("not a context file", stub) >= 0 && fclose(stub) == 0);
    CHECK(h3_motion_context_plan("/tmp/h3-motion-context-test.mp4", lh, lw,
                                 &context, error, sizeof(error)));
    CHECK(!context.from_latents && context.audio_t == 37 &&
          context.audio_offset == 0.0);
    remove("/tmp/h3-motion-context-test.mp4");
    /* Trim: 22 of 107 frames at 32 kHz cuts 29333 samples and sizes the rest
     * to 85 frames (113333 samples), zero-padding a short audio grid. */
    int samples = 178 * 800;
    float *pcm = malloc((size_t)2 * samples * sizeof(*pcm));
    for (int index = 0; index < 2 * samples; index++) pcm[index] = 1.0f;
    float *trimmed = NULL;
    int trimmed_samples = 0;
    CHECK(h3_motion_context_trim_pcm(pcm, 2, samples, 32000, 107, 22,
                                     &trimmed, &trimmed_samples));
    CHECK(trimmed_samples == 113333);
    CHECK(trimmed[0] == 1.0f && trimmed[113066] == 1.0f &&
          trimmed[113067] == 0.0f && trimmed[113333] == 1.0f &&
          trimmed[2 * 113333 - 1] == 0.0f);
    CHECK(!h3_motion_context_trim_pcm(pcm, 2, samples, 32000, 22, 22,
                                      &trimmed, &trimmed_samples));
    free(trimmed); free(pcm); free(audio_rows); free(audio_tail); free(rows);
    free(tail); free(video); free(audio);
}

static void test_layout_ref2va(void) {
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 8287);
    const size_t bounds[][2] = {
        {0, 192}, {192, 288}, {288, 384}, {384, 1056},
        {1056, 1216}, {1216, 1402}, {1402, 8287}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_AUDIO, H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 7);
    CHECK(layout.img_cond_rows == 768);
    CHECK(layout.audio_cond_rows == 256);
    double sums[3], weighted[3];
    position_checksums(&layout, sums, weighted);
    CHECK(close_enough(sums[0], 2818905.0, 1e-5));
    CHECK(close_enough(weighted[0], 12788836714.0, 1e-3));
    CHECK(close_enough(sums[1], 115719.96684277552, 1e-7));
    CHECK(close_enough(sums[2], 122360.11344760543, 1e-7));
    h3_layout_free(&layout);
}

static void test_layout_motion_context_refs(void) {
    /* The pinned head and its audio window sit on the target timeline, which
     * starts after the reference spans (1 + 48 + 80 steps here), while the
     * references themselves keep their slots right after the text. */
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           NULL, 0, references, 3, 7, 37, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 11196);
    const size_t bounds[][2] = {
        {0, 192}, {192, 3027}, {3027, 3101}, {3101, 3197}, {3197, 3293},
        {3293, 3965}, {3965, 4125}, {4125, 4311}, {4311, 11196}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_AUDIO,
        H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 9);
    CHECK(layout.img_cond_rows == 2835 + 768);
    CHECK(layout.audio_cond_rows == 74 + 256);
    CHECK(layout.img_target_rows == 6885 && layout.audio_target_rows == 186);
    CHECK(layout.positions[192].t == 321.0);
    CHECK(layout.positions[3101].t == 192.0);
    CHECK(layout.positions[4125].t == 321.0);
    int same = 1;
    for (size_t row = 0; row < 2835; row++) {
        const h3_position *pinned = &layout.positions[192 + row];
        const h3_position *target = &layout.positions[4311 + row];
        if (pinned->t != target->t || pinned->h != target->h ||
            pinned->w != target->w) same = 0;
    }
    CHECK(same);
    same = 1;
    for (size_t step = 0; step < 37; step++) {
        for (int stereo = 0; stereo < 2; stereo++) {
            const h3_position *pinned =
                &layout.positions[3027 + (size_t)stereo * 37 + step];
            const h3_position *target =
                &layout.positions[4125 + (size_t)stereo * 93 + step];
            if (pinned->t != target->t || pinned->w != target->w) same = 0;
        }
    }
    CHECK(same);
    h3_layout_free(&layout);
}

static void test_layout_hybrid(void) {
    /* Frame anchors with references: the anchors lead the sequence as in
     * the FL2VA layout but are timed from the target origin, after the
     * reference spans, where the target rows they stand in for sit. */
    int keyframes[] = {0, 55};
    h3_layout_ref references[] = {
        {H3_LAYOUT_REF_IMAGE, 0, 16, 24, 0},
        {H3_LAYOUT_REF_VIDEO, 7, 16, 24, 48},
        {H3_LAYOUT_REF_AUDIO, 0, 0, 0, 80}
    };
    h3_layout_spec spec = {192, 17, 30, 54, 93, 56,
                           keyframes, 2, references, 3, 0, 0, 0.0};
    h3_layout layout;
    char error[256];
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.seq_len == 9097);
    const size_t bounds[][2] = {
        {0, 192}, {192, 597}, {597, 1002}, {1002, 1098}, {1098, 1194},
        {1194, 1866}, {1866, 2026}, {2026, 2212}, {2212, 9097}
    };
    const h3_segment_kind kinds[] = {
        H3_SEG_TEXT, H3_SEG_COND, H3_SEG_COND, H3_SEG_REF_IMAGE,
        H3_SEG_REF_AUDIO, H3_SEG_REF_IMAGE, H3_SEG_REF_AUDIO, H3_SEG_AUDIO,
        H3_SEG_VIDEO
    };
    check_segments(&layout, bounds, kinds, 9);
    CHECK(layout.img_cond_rows == 810 + 768);
    CHECK(layout.audio_cond_rows == 256);
    CHECK(layout.positions[192].t == layout.positions[2212].t);
    CHECK(layout.positions[192].t == 321.0);
    CHECK(close_enough(layout.positions[597].t, 321.0 + 55.0 * 5.0 / 3.0, 1e-9));
    CHECK(layout.positions[1002].t == 192.0);
    h3_layout_free(&layout);
    /* A pinned head still replaces the first-frame anchor. */
    spec.head_steps = 7;
    spec.head_audio_t = 37;
    CHECK(!h3_layout_build(&spec, &layout, error, sizeof(error)));
    spec.keyframes = &keyframes[1];
    spec.keyframe_count = 1;
    CHECK(h3_layout_build(&spec, &layout, error, sizeof(error)));
    CHECK(layout.img_cond_rows == 2835 + 405 + 768);
    CHECK(layout.segments[1].kind == H3_SEG_COND &&
          layout.segments[2].kind == H3_SEG_REF_AUDIO &&
          layout.segments[3].kind == H3_SEG_COND &&
          layout.segments[4].kind == H3_SEG_REF_IMAGE);
    h3_layout_free(&layout);
}

static void write_all(int descriptor, const void *data, size_t size) {
    const unsigned char *bytes = data;
    while (size) {
        ssize_t written = write(descriptor, bytes, size);
        CHECK(written > 0);
        bytes += (size_t)written;
        size -= (size_t)written;
    }
}

static void test_safetensors(void) {
    char path[] = "/tmp/h3_safetensors_XXXXXX";
    int descriptor = mkstemp(path);
    CHECK(descriptor >= 0);
    const char header_json[] =
        "{\"scalar\":{\"dtype\":\"BF16\",\"shape\":[],\"data_offsets\":[24,26]},"
        "\"x\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]}}";
    uint64_t length = sizeof(header_json) - 1;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; index++) prefix[index] = (unsigned char)(length >> (8 * index));
    unsigned char payload[26] = {0};
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, header_json, (size_t)length);
    write_all(descriptor, payload, sizeof(payload));
    CHECK(close(descriptor) == 0);

    h3_st_header header;
    char error[256];
    CHECK(h3_st_read_header(path, &header, error, sizeof(error)));
    CHECK(header.tensor_count == 2);
    const h3_st_tensor *x = h3_st_find(&header, "x");
    CHECK(x && x->dtype == H3_DTYPE_F32 && x->ndim == 2);
    CHECK(x->shape[0] == 2 && x->shape[1] == 3);
    CHECK(x->data_end - x->data_begin == 24);
    CHECK(h3_st_tensor_elements(x) == 6);
    unsigned char readback[24];
    CHECK(h3_st_read_data(&header, x, readback, sizeof(readback), error,
                          sizeof(error)));
    for (size_t index = 0; index < sizeof(readback); index++) {
        CHECK(readback[index] == 0);
    }
    CHECK(!h3_st_read_data(&header, x, readback, sizeof(readback) - 1, error,
                           sizeof(error)));
    const h3_st_tensor *scalar = h3_st_find(&header, "scalar");
    CHECK(scalar && scalar->dtype == H3_DTYPE_BF16 && scalar->ndim == 0);
    CHECK(scalar->data_end - scalar->data_begin == 2);
    CHECK(h3_st_tensor_elements(scalar) == 1);
    h3_st_free_header(&header);
    CHECK(unlink(path) == 0);

    char overlap_path[] = "/tmp/h3_safetensors_overlap_XXXXXX";
    descriptor = mkstemp(overlap_path);
    CHECK(descriptor >= 0);
    const char overlap_json[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]},"
        "\"y\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[2,6]}}";
    length = sizeof(overlap_json) - 1;
    for (unsigned index = 0; index < 8; index++)
        prefix[index] = (unsigned char)(length >> (8 * index));
    unsigned char overlap_payload[6] = {0};
    write_all(descriptor, prefix, sizeof(prefix));
    write_all(descriptor, overlap_json, (size_t)length);
    write_all(descriptor, overlap_payload, sizeof(overlap_payload));
    CHECK(close(descriptor) == 0);
    CHECK(!h3_st_read_header(overlap_path, &header, error, sizeof(error)));
    CHECK(unlink(overlap_path) == 0);
}

static void write_file(const char *path, const void *data, size_t size) {
    int descriptor = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    CHECK(descriptor >= 0);
    write_all(descriptor, data, size);
    CHECK(close(descriptor) == 0);
}

static void set_mtime(const char *path, long seconds) {
    struct timeval times[2] = {{seconds, 0}, {seconds, 0}};
    CHECK(utimes(path, times) == 0);
}

static int exists(const char *path) {
    return access(path, F_OK) == 0;
}

static void test_dit_cache(void) {
    char root[] = "/tmp/h3_dit_cache_XXXXXX";
    CHECK(mkdtemp(root) != NULL);
    char checkpoint[256], shard[320], notes[320], shader[256];
    char directory[256];
    snprintf(checkpoint, sizeof(checkpoint), "%s/transformer", root);
    snprintf(shard, sizeof(shard), "%s/model-00001.safetensors", checkpoint);
    snprintf(notes, sizeof(notes), "%s/README.md", checkpoint);
    snprintf(shader, sizeof(shader), "%s/h3_shaders.metal", root);
    snprintf(directory, sizeof(directory), "%s/cache", root);
    CHECK(mkdir(checkpoint, 0700) == 0);
    unsigned char shard_bytes[16] = {1, 2, 3};
    write_file(shard, shard_bytes, sizeof(shard_bytes));
    set_mtime(shard, 1700000000);
    const char kernel_source[] = "kernel void h3_test() {}\n";
    write_file(shader, kernel_source, sizeof(kernel_source) - 1);
    const char *m5 = "Apple M5 Ultra|tensor-ops=1:3";

    /* The cache directory is created private. */
    h3_dit_cache cache;
    CHECK(h3_dit_cache_init(&cache, directory, checkpoint, m5, shader));
    struct stat info;
    CHECK(stat(directory, &info) == 0 && S_ISDIR(info.st_mode));
    CHECK((info.st_mode & 0777) == 0700);
    CHECK(!h3_dit_cache_init(&cache, shard, checkpoint, m5, shader));
    CHECK(h3_dit_cache_init(&cache, directory, checkpoint, m5, shader));

    /* Checkpoint fingerprint: stable, sensitive to shard mtime, size and
     * names, blind to non-shard files. */
    uint64_t fingerprint = h3_dit_cache_checkpoint_fingerprint(checkpoint);
    CHECK(fingerprint == cache.checkpoint);
    CHECK(fingerprint == h3_dit_cache_checkpoint_fingerprint(checkpoint));
    set_mtime(shard, 1700000001);
    uint64_t touched = h3_dit_cache_checkpoint_fingerprint(checkpoint);
    CHECK(touched != fingerprint);
    set_mtime(shard, 1700000000);
    CHECK(h3_dit_cache_checkpoint_fingerprint(checkpoint) == fingerprint);
    unsigned char grown[17] = {1, 2, 3};
    write_file(shard, grown, sizeof(grown));
    set_mtime(shard, 1700000000);
    uint64_t resized = h3_dit_cache_checkpoint_fingerprint(checkpoint);
    CHECK(resized != fingerprint && resized != touched);
    write_file(shard, shard_bytes, sizeof(shard_bytes));
    set_mtime(shard, 1700000000);
    CHECK(h3_dit_cache_checkpoint_fingerprint(checkpoint) == fingerprint);
    write_file(notes, "notes", 5);
    CHECK(h3_dit_cache_checkpoint_fingerprint(checkpoint) == fingerprint);
    CHECK(unlink(notes) == 0);
    char renamed[320];
    snprintf(renamed, sizeof(renamed), "%s/model-00002.safetensors",
             checkpoint);
    CHECK(rename(shard, renamed) == 0);
    CHECK(h3_dit_cache_checkpoint_fingerprint(checkpoint) != fingerprint);
    CHECK(rename(renamed, shard) == 0);
    CHECK(h3_dit_cache_checkpoint_fingerprint(checkpoint) == fingerprint);

    /* The kernel variant follows the device/TensorOps mode and the shader
     * content, not the shader path's mtime. */
    h3_dit_cache other;
    CHECK(h3_dit_cache_init(&other, directory, checkpoint,
                            "Apple M3 Max|tensor-ops=0:0", shader));
    CHECK(other.checkpoint == cache.checkpoint);
    CHECK(other.source == cache.source);
    CHECK(other.variant != cache.variant);
    set_mtime(shader, 1600000000);
    CHECK(h3_dit_cache_init(&other, directory, checkpoint, m5, shader));
    CHECK(other.variant == cache.variant);
    write_file(shader, "kernel void h3_other() {}\n", 26);
    CHECK(h3_dit_cache_init(&other, directory, checkpoint, m5, shader));
    CHECK(other.variant != cache.variant);

    /* int8 keys: one per block (layer) and stored-matrix set. */
    const unsigned all = 0xFu;
    uint64_t key = h3_dit_cache_int8_key(&cache, 7, all);
    CHECK(key == h3_dit_cache_int8_key(&cache, 7, all));
    CHECK(key != h3_dit_cache_int8_key(&cache, 8, all));
    CHECK(key != h3_dit_cache_int8_key(&cache, 7, 0xCu));
    other = cache;
    other.checkpoint ^= 1;
    CHECK(key != h3_dit_cache_int8_key(&other, 7, all));
    other = cache;
    other.variant ^= 1;
    CHECK(key != h3_dit_cache_int8_key(&other, 7, all));

    /* AdaLN keys: steps, every used sigma bit and both condition flags. */
    h3_sigma_schedule standard, turbo, nudged;
    CHECK(h3_serving_schedule_build(20, &standard));
    CHECK(h3_serving_schedule_build(6, &turbo));
    uint64_t adaln = h3_dit_cache_adaln_key(&cache, &standard, 1, 0);
    CHECK(adaln == h3_dit_cache_adaln_key(&cache, &standard, 1, 0));
    CHECK(adaln != h3_dit_cache_adaln_key(&cache, &turbo, 1, 0));
    CHECK(adaln != h3_dit_cache_adaln_key(&cache, &standard, 0, 0));
    CHECK(adaln != h3_dit_cache_adaln_key(&cache, &standard, 1, 1));
    CHECK(adaln != h3_dit_cache_int8_key(&cache, 7, all));
    nudged = standard;
    nudged.video[3] = nextafterf(nudged.video[3], 2.0f);
    CHECK(adaln != h3_dit_cache_adaln_key(&cache, &nudged, 1, 0));
    nudged = standard;
    nudged.audio[19] = nextafterf(nudged.audio[19], 2.0f);
    CHECK(adaln != h3_dit_cache_adaln_key(&cache, &nudged, 1, 0));
    nudged = standard;
    nudged.video[25] = 0.5f;
    CHECK(adaln == h3_dit_cache_adaln_key(&cache, &nudged, 1, 0));
    other = cache;
    other.checkpoint ^= 1;
    CHECK(adaln != h3_dit_cache_adaln_key(&other, &standard, 1, 0));

    char path[1200], adaln_path[1200];
    CHECK(h3_dit_cache_int8_path(&cache, 7, key, path, sizeof(path)));
    CHECK(!strncmp(path, directory, strlen(directory)));
    CHECK(strstr(path, "-int8-b07-") != NULL);
    CHECK(!h3_dit_cache_int8_path(&cache, 7, key, path, 16));
    CHECK(h3_dit_cache_int8_path(&cache, 7, key, path, sizeof(path)));
    CHECK(h3_dit_cache_adaln_path(&cache, 20, adaln, adaln_path,
                                  sizeof(adaln_path)));
    CHECK(strstr(adaln_path, "-adaln-s20-") != NULL);

    /* Layout: page-aligned payloads and file end. */
    const uint64_t sizes[3] = {100, 20000, 7};
    h3_dit_cache_header header, expected;
    CHECK(h3_dit_cache_header_init(&header, H3_DIT_CACHE_INT8, key, sizes,
                                   3));
    CHECK(header.entry_count == 3);
    CHECK(header.entries[0].offset == H3_DIT_CACHE_ALIGN);
    CHECK(header.entries[1].offset == 2 * H3_DIT_CACHE_ALIGN);
    CHECK(header.entries[2].offset == 4 * H3_DIT_CACHE_ALIGN);
    CHECK(header.file_bytes == 5 * H3_DIT_CACHE_ALIGN);
    CHECK(!h3_dit_cache_header_init(&expected, H3_DIT_CACHE_INT8, key, sizes,
                                    0));
    const uint64_t empty[2] = {100, 0};
    CHECK(!h3_dit_cache_header_init(&expected, H3_DIT_CACHE_INT8, key, empty,
                                    2));

    /* Atomic private write, then validation against the reader's
     * expectation. */
    unsigned char first[100], second[20000], third[7];
    for (size_t index = 0; index < sizeof(first); index++)
        first[index] = (unsigned char)index;
    for (size_t index = 0; index < sizeof(second); index++)
        second[index] = (unsigned char)(index * 7 + 1);
    memset(third, 0xA5, sizeof(third));
    const void *payloads[3] = {first, second, third};
    char error[256];
    CHECK(!h3_dit_cache_valid(path, &header));
    CHECK(h3_dit_cache_write(path, &header, payloads, error, sizeof(error)));
    CHECK(h3_dit_cache_valid(path, &header));
    CHECK(stat(path, &info) == 0);
    CHECK((info.st_mode & 0777) == 0600);
    CHECK((uint64_t)info.st_size == header.file_bytes);
    char temp[1300];
    snprintf(temp, sizeof(temp), "%s.tmp.%d", path, (int)getpid());
    CHECK(!exists(temp));
    int descriptor = open(path, O_RDONLY);
    CHECK(descriptor >= 0);
    unsigned char readback[20000];
    CHECK(pread(descriptor, readback, sizeof(readback),
                (off_t)header.entries[1].offset) == (ssize_t)sizeof(readback));
    CHECK(!memcmp(readback, second, sizeof(second)));
    CHECK(close(descriptor) == 0);

    expected = header;
    expected.key ^= 1;
    CHECK(!h3_dit_cache_valid(path, &expected));
    const uint64_t resized_entry[3] = {100, 20000, 8};
    CHECK(h3_dit_cache_header_init(&expected, H3_DIT_CACHE_INT8, key,
                                   resized_entry, 3));
    CHECK(!h3_dit_cache_valid(path, &expected));
    CHECK(h3_dit_cache_header_init(&expected, H3_DIT_CACHE_ADALN, key, sizes,
                                   3));
    CHECK(!h3_dit_cache_valid(path, &expected));

    /* A corrupt or truncated file is rejected. */
    descriptor = open(path, O_RDWR);
    CHECK(descriptor >= 0);
    const size_t corrupt_offsets[] = {0, 8,
        offsetof(h3_dit_cache_header, key),
        offsetof(h3_dit_cache_header, entries) + 16};
    for (size_t index = 0;
         index < sizeof(corrupt_offsets) / sizeof(*corrupt_offsets); index++) {
        unsigned char byte = 0;
        off_t at = (off_t)corrupt_offsets[index];
        CHECK(pread(descriptor, &byte, 1, at) == 1);
        unsigned char flipped = (unsigned char)(byte ^ 0x40);
        CHECK(pwrite(descriptor, &flipped, 1, at) == 1);
        CHECK(!h3_dit_cache_valid(path, &header));
        CHECK(pwrite(descriptor, &byte, 1, at) == 1);
        CHECK(h3_dit_cache_valid(path, &header));
    }
    CHECK(ftruncate(descriptor, (off_t)header.file_bytes - 1) == 0);
    CHECK(!h3_dit_cache_valid(path, &header));
    CHECK(ftruncate(descriptor, 64) == 0);
    CHECK(!h3_dit_cache_valid(path, &header));
    CHECK(close(descriptor) == 0);
    CHECK(h3_dit_cache_write(path, &header, payloads, error, sizeof(error)));
    CHECK(h3_dit_cache_valid(path, &header));

    /* Pruning removes superseded files of this block and temporaries of dead
     * writers only. */
    char stale[1200], other_block[1200], dead_temp[1300], live_temp[1300];
    CHECK(h3_dit_cache_int8_path(&cache, 7, key ^ 1, stale, sizeof(stale)));
    CHECK(h3_dit_cache_int8_path(&cache, 8, key, other_block,
                                 sizeof(other_block)));
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) _exit(0);
    CHECK(waitpid(child, NULL, 0) == child);
    snprintf(dead_temp, sizeof(dead_temp), "%s.tmp.%d", stale, (int)child);
    snprintf(live_temp, sizeof(live_temp), "%s.tmp.%d", stale,
             (int)getpid());
    write_file(stale, "x", 1);
    write_file(other_block, "x", 1);
    write_file(dead_temp, "x", 1);
    write_file(live_temp, "x", 1);
    write_file(adaln_path, "x", 1);
    h3_dit_cache_prune_int8(&cache, 7, path);
    CHECK(exists(path));
    CHECK(!exists(stale));
    CHECK(!exists(dead_temp));
    CHECK(exists(live_temp));
    CHECK(exists(other_block));
    CHECK(exists(adaln_path));

    CHECK(unlink(path) == 0);
    CHECK(unlink(live_temp) == 0);
    CHECK(unlink(other_block) == 0);
    CHECK(unlink(adaln_path) == 0);
    CHECK(rmdir(directory) == 0);
    CHECK(unlink(shard) == 0);
    CHECK(rmdir(checkpoint) == 0);
    CHECK(unlink(shader) == 0);
    CHECK(rmdir(root) == 0);
}

static void test_rng_and_solver(void) {
    h3_rng a, b;
    h3_rng_seed(&a, 42);
    h3_rng_seed(&b, 42);
    for (int index = 0; index < 64; index++) CHECK(h3_rng_u32(&a) == h3_rng_u32(&b));
    h3_rng_seed(&a, 7);
    float values[1024];
    h3_rng_fill_normal(&a, values, 1024);
    double sum = 0.0;
    for (size_t index = 0; index < 1024; index++) {
        CHECK(isfinite(values[index]));
        sum += values[index];
    }
    CHECK(fabs(sum / 1024.0) < 0.15);

    const float sigmas[] = {1.0f, 0.75f, 0.25f, 0.0f};
    float sample[] = {0.0f};
    float denoised[] = {1.0f};
    float old[] = {0.5f};
    float output[1];
    CHECK(h3_res_step(output, sample, denoised, NULL, 1, sigmas, 0, 3));
    CHECK(close_enough(output[0], 0.25, 1e-7));
    CHECK(h3_res_step(output, sample, denoised, old, 1, sigmas, 1, 3));
    CHECK(isfinite(output[0]));
    float velocity[] = {2.0f, -4.0f};
    float euler[] = {1.0f, 3.0f};
    CHECK(h3_euler_velocity_step(euler, velocity, 2, 0.75f, 0.25f));
    CHECK(euler[0] == 2.0f && euler[1] == 1.0f);
    CHECK(!h3_euler_velocity_step(euler, velocity, 2, 0.25f, 0.25f));
}

static void test_rgb_resize(void) {
    const uint8_t constant[] = {
        17, 33, 201, 17, 33, 201,
        17, 33, 201, 17, 33, 201
    };
    uint8_t *identity = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 2, 2, &identity));
    CHECK(identity != NULL);
    CHECK(memcmp(identity, constant, sizeof(constant)) == 0);
    free(identity);

    uint8_t *upscaled = NULL;
    CHECK(h3_resize_rgb24_high_quality(constant, 1, 2, 2, 8, 8, &upscaled));
    CHECK(upscaled != NULL);
    for (size_t pixel = 0; pixel < 64; pixel++) {
        CHECK(upscaled[3 * pixel] == 17);
        CHECK(upscaled[3 * pixel + 1] == 33);
        CHECK(upscaled[3 * pixel + 2] == 201);
    }
    free(upscaled);
    CHECK(!h3_resize_rgb24_high_quality(NULL, 1, 2, 2, 8, 8, &upscaled));
}

static void test_dit_row_conversions(void) {
    enum { C = 3, T = 2, H = 4, W = 6, VIDEO = C * T * H * W };
    float latent[VIDEO], rows[VIDEO], roundtrip[VIDEO];
    for (size_t index = 0; index < VIDEO; index++) latent[index] = (float)index;
    CHECK(h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO));
    /* First row is pixel patch (h=0..1,w=0..1), channel-major inside it. */
    const float first[] = {0, 1, 6, 7, 48, 49, 54, 55, 96, 97, 102, 103};
    for (size_t index = 0; index < sizeof(first) / sizeof(*first); index++)
        CHECK(rows[index] == first[index]);
    CHECK(h3_dit_unpatchify_video(rows, C, T, H, W, roundtrip, VIDEO));
    CHECK(memcmp(latent, roundtrip, sizeof(latent)) == 0);
    CHECK(!h3_dit_patchify_video(latent, C, T, H, W, rows, VIDEO - 1));

    enum { AC = 4, AT = 3, AUDIO = AC * 2 * AT };
    float audio[AUDIO], packed[AUDIO], unpacked[AUDIO];
    for (size_t index = 0; index < AUDIO; index++) audio[index] = (float)index;
    CHECK(h3_dit_pack_audio(audio, AC, AT, packed, AUDIO));
    const float expected[] = {0, 6, 12, 18, 1, 7, 13, 19,
                              2, 8, 14, 20, 3, 9, 15, 21,
                              4, 10, 16, 22, 5, 11, 17, 23};
    CHECK(memcmp(packed, expected, sizeof(expected)) == 0);
    CHECK(h3_dit_unpack_audio(packed, AC, AT, unpacked, AUDIO));
    CHECK(memcmp(audio, unpacked, sizeof(audio)) == 0);
}

static void test_metal_probe(void) {
    h3_device_info info;
    char error[256];
    CHECK(h3_metal_probe(&info, error, sizeof(error)));
    CHECK(info.name[0] != '\0');
    CHECK(info.physical_memory >= UINT64_C(8) * 1024 * 1024 * 1024);
    CHECK(info.max_buffer_length > 0);
    CHECK(info.apple_gpu_family > 0);
}

static void test_terminal_zoom(void) {
    int width = 0, height = 0;
    CHECK(h3_terminal_set_zoom(2));
    CHECK(h3_terminal_display_dimensions(512, 288, &width, &height));
    CHECK(width == 1024 && height == 576);
    CHECK(h3_terminal_set_zoom(1));
    CHECK(h3_terminal_display_dimensions(512, 288, &width, &height));
    CHECK(width == 512 && height == 288);
    CHECK(!h3_terminal_set_zoom(0));
    CHECK(h3_terminal_set_zoom(2));
    CHECK(!h3_terminal_display_dimensions(INT32_MAX, 1, &width, &height));
}

static int reflect_index(int coordinate, int length) {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= length) return 2 * length - coordinate - 2;
    return coordinate;
}

static float synthetic_value(uint32_t index, uint32_t salt) {
    uint32_t state = index * 2654435761u + salt * 40503u + 12345u;
    state ^= state >> 13;
    return (float)((int)(state % 2001u) - 1000) / 337.0f;
}

/* A one-frame causal Conv3d (kernel-1 zero front frames, reflect spatial
 * padding, any temporal stride) must equal the Conv2d over its last temporal
 * tap exactly: the dropped taps only ever multiply the zero front frames. */
static void check_single_frame_conv(uint32_t kernel, uint32_t stride_t,
                                    uint32_t stride, uint32_t pad_before,
                                    uint32_t pad_after) {
    enum { IN = 3, OUT = 4, H = 6, W = 7 };
    uint32_t front = kernel - 1;
    uint32_t padded_d = 1 + front;
    uint32_t padded_h = H + pad_before + pad_after;
    uint32_t padded_w = W + pad_before + pad_after;
    CHECK((padded_d - kernel) / stride_t + 1 == 1);
    uint32_t out_h = (padded_h - kernel) / stride + 1;
    uint32_t out_w = (padded_w - kernel) / stride + 1;
    size_t taps = (size_t)kernel * kernel * kernel;
    float *input = malloc((size_t)H * W * IN * sizeof(*input));
    float *weight = malloc((size_t)OUT * IN * taps * sizeof(*weight));
    float *planar = malloc((size_t)OUT * IN * kernel * kernel *
                           sizeof(*planar));
    float *volume = calloc((size_t)padded_d * padded_h * padded_w * IN,
                           sizeof(*volume));
    float bias[OUT];
    CHECK(input && weight && planar && volume);
    for (uint32_t index = 0; index < (uint32_t)(H * W * IN); index++)
        input[index] = synthetic_value(index, 1);
    for (uint32_t index = 0; index < (uint32_t)(OUT * IN * taps); index++)
        weight[index] = synthetic_value(index, 2);
    for (uint32_t index = 0; index < OUT; index++)
        bias[index] = synthetic_value(index, 3);
    h3_video_encoder_slice_tap_f32(weight, OUT, IN, kernel, kernel - 1, planar);
    for (uint32_t pair = 0; pair < OUT * IN; pair++)
        for (uint32_t tap = 0; tap < kernel * kernel; tap++)
            CHECK(planar[pair * kernel * kernel + tap] ==
                  weight[(pair * kernel + kernel - 1) * kernel * kernel + tap]);
    /* The real frame follows `front` zero frames, as h3_vae_encoder_pad_f32. */
    for (uint32_t y = 0; y < padded_h; y++)
        for (uint32_t x = 0; x < padded_w; x++)
            for (uint32_t channel = 0; channel < IN; channel++) {
                int source_y = reflect_index((int)y - (int)pad_before, H);
                int source_x = reflect_index((int)x - (int)pad_before, W);
                volume[(((size_t)front * padded_h + y) * padded_w + x) * IN +
                       channel] =
                    input[((size_t)source_y * W + (size_t)source_x) * IN +
                          channel];
            }
    const float *plane = volume + (size_t)front * padded_h * padded_w * IN;
    for (uint32_t y = 0; y < out_h; y++)
        for (uint32_t x = 0; x < out_w; x++)
            for (uint32_t output = 0; output < OUT; output++) {
                float full = bias[output], sliced = bias[output];
                for (uint32_t kd = 0; kd < kernel; kd++)
                    for (uint32_t kh = 0; kh < kernel; kh++)
                        for (uint32_t kw = 0; kw < kernel; kw++)
                            for (uint32_t channel = 0; channel < IN; channel++)
                                full += volume[(((size_t)kd * padded_h +
                                    y * stride + kh) * padded_w +
                                    x * stride + kw) * IN + channel] *
                                    weight[((((size_t)output * IN + channel) *
                                    kernel + kd) * kernel + kh) * kernel + kw];
                for (uint32_t kh = 0; kh < kernel; kh++)
                    for (uint32_t kw = 0; kw < kernel; kw++)
                        for (uint32_t channel = 0; channel < IN; channel++)
                            sliced += plane[(((size_t)y * stride + kh) *
                                padded_w + x * stride + kw) * IN + channel] *
                                planar[(((size_t)output * IN + channel) *
                                kernel + kh) * kernel + kw];
                CHECK(full == sliced);
            }
    free(input);
    free(weight);
    free(planar);
    free(volume);
}

static void test_video_encoder_single_frame_conv(void) {
    check_single_frame_conv(3, 1, 1, 1, 1);
    check_single_frame_conv(3, 2, 2, 0, 1);
    check_single_frame_conv(3, 1, 2, 0, 1);
    check_single_frame_conv(1, 1, 1, 0, 0);
}

static void test_video_vae_assembly(void) {
    char error[256] = {0};
    int ok = h3_video_vae_assembly_selftest(error, sizeof(error));
    if (!ok) fprintf(stderr, "VAE assembly: %s\n", error);
    CHECK(ok);
}

static void *read_file(const char *path, size_t *size) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    fseek(file, 0, SEEK_END);
    long end = ftell(file);
    fseek(file, 0, SEEK_SET);
    void *data = end > 0 ? malloc((size_t)end) : NULL;
    if (data && fread(data, 1, (size_t)end, file) != (size_t)end) {
        free(data);
        data = NULL;
    }
    fclose(file);
    *size = data ? (size_t)end : 0;
    return data;
}

/* Needs an FFmpeg with libx264; runs only when H3_FFMPEG names one. The
 * streamed writer, fed in uneven runs with pauses, must produce the same
 * file bytes as the whole-buffer writer, and abort must leave no file. */
static void test_ffmpeg_stream(void) {
    const char *ffmpeg = getenv("H3_FFMPEG");
    if (!ffmpeg || !*ffmpeg) {
        printf("skip: FFmpeg stream test (set H3_FFMPEG)\n");
        return;
    }
    enum { FRAMES = 22, W = 64, H = 48, RATE = 32000, SAMPLES = RATE };
    size_t frame_bytes = (size_t)W * H * 3;
    uint8_t *frames = malloc(FRAMES * frame_bytes);
    float *pcm = malloc(2 * SAMPLES * sizeof(*pcm));
    CHECK(frames && pcm);
    for (size_t index = 0; index < FRAMES * frame_bytes; index++)
        frames[index] = (uint8_t)((index * 2654435761u) >> 24);
    for (size_t index = 0; index < 2 * SAMPLES; index++)
        pcm[index] = sinf((float)index * 0.01f) * 0.25f;
    char directory[] = "/tmp/h3-ffmpeg-test-XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    char whole[256], streamed[256], aborted[256];
    snprintf(whole, sizeof(whole), "%s/whole.mp4", directory);
    snprintf(streamed, sizeof(streamed), "%s/streamed.mp4", directory);
    snprintf(aborted, sizeof(aborted), "%s/aborted.mp4", directory);
    char error[256];
    CHECK(h3_ffmpeg_write_av_rgb24_f32(whole, frames, FRAMES, W, H, 24, pcm,
                                       SAMPLES, 2, RATE, error,
                                       sizeof(error)));
    h3_ffmpeg_av_stream *stream = h3_ffmpeg_av_stream_open(
        streamed, frames, FRAMES, W, H, 24, pcm, SAMPLES, 2, RATE, error,
        sizeof(error));
    CHECK(stream != NULL);
    const int runs[] = {17, 17, 22};
    for (size_t index = 0; index < sizeof(runs) / sizeof(*runs); index++) {
        usleep(50000);
        h3_ffmpeg_av_stream_ready(stream, runs[index]);
    }
    CHECK(h3_ffmpeg_av_stream_finish(stream, error, sizeof(error)));
    size_t whole_size = 0, streamed_size = 0;
    uint8_t *whole_bytes = read_file(whole, &whole_size);
    uint8_t *streamed_bytes = read_file(streamed, &streamed_size);
    CHECK(whole_bytes && streamed_bytes && whole_size == streamed_size);
    CHECK(memcmp(whole_bytes, streamed_bytes, whole_size) == 0);
    stream = h3_ffmpeg_av_stream_open(aborted, frames, FRAMES, W, H, 24, pcm,
                                      SAMPLES, 2, RATE, error, sizeof(error));
    CHECK(stream != NULL);
    h3_ffmpeg_av_stream_ready(stream, 17);
    h3_ffmpeg_av_stream_abort(stream);
    CHECK(!exists(aborted));
    unlink(whole);
    unlink(streamed);
    CHECK(rmdir(directory) == 0);   /* so no partial file was left behind */
    free(whole_bytes);
    free(streamed_bytes);
    free(frames);
    free(pcm);
}

/* The tail reader keeps the last N decoded frames of a stream it reads
 * once; checked against the bounded reader on a synthetic 30-frame clip. */
static void test_ffmpeg_video_tail(void) {
    const char *ffmpeg = getenv("H3_FFMPEG");
    if (!ffmpeg || !*ffmpeg) {
        printf("skip: FFmpeg video tail test (set H3_FFMPEG)\n");
        return;
    }
    enum { FRAMES = 30, W = 64, H = 48, RATE = 32000, SAMPLES = RATE };
    size_t frame_bytes = (size_t)W * H * 3;
    uint8_t *frames = malloc(FRAMES * frame_bytes);
    float *pcm = calloc(2 * SAMPLES, sizeof(*pcm));
    CHECK(frames && pcm);
    /* Each frame a flat shade so lossy encoding keeps them apart. */
    for (int frame = 0; frame < FRAMES; frame++)
        memset(frames + (size_t)frame * frame_bytes, 16 + frame * 7, frame_bytes);
    char path[] = "/tmp/h3-ffmpeg-tail-test.mp4";
    char error[256];
    CHECK(h3_ffmpeg_write_av_rgb24_f32(path, frames, FRAMES, W, H, 24, pcm,
                                       SAMPLES, 2, RATE, error, sizeof(error)));
    float *whole = NULL, *tail = NULL;
    int whole_frames = 0, tail_frames = 0, total_frames = 0;
    CHECK(h3_ffmpeg_read_video_f32(path, W, H, FRAMES, &whole, &whole_frames,
                                   error, sizeof(error)));
    CHECK(h3_ffmpeg_read_video_tail_f32(path, W, H, 22, &tail, &tail_frames,
                                        &total_frames, error, sizeof(error)));
    /* The bounded reader snaps 30 frames down to 22 (frames 0..21); the
     * tail is frames 8..29, so the 14 shared frames must match exactly. */
    CHECK(tail_frames == 22 && whole_frames == 22 && total_frames == FRAMES);
    size_t area = (size_t)W * H;
    int same = whole && tail;
    for (int channel = 0; same && channel < 3; channel++)
        for (int t = 0; same && t < 14; t++)
            if (memcmp(tail + ((size_t)channel * 22 + t) * area,
                       whole + ((size_t)channel * 22 + 8 + t) * area,
                       area * sizeof(float)) != 0) same = 0;
    CHECK(same);
    /* Shades rise 7 levels a frame: the last tail frame is the brightest. */
    CHECK(tail[21 * area] > tail[0] + 100.0f / 255.0f);
    float *more = NULL;
    int more_frames = 0;
    CHECK(h3_ffmpeg_read_video_tail_f32(path, W, H, 40, &more, &more_frames,
                                        NULL, error, sizeof(error)));
    CHECK(more_frames == FRAMES);
    unlink(path);
    free(whole); free(tail); free(more); free(frames); free(pcm);
}

int main(void) {
    test_temporal_and_canvas();
    test_schedule();
    test_dit_reuse_schedule();
    test_layout_tiny();
    test_layout_fl2va();
    test_layout_motion_context();
    test_motion_context_file();
    test_layout_ref2va();
    test_layout_motion_context_refs();
    test_layout_hybrid();
    test_safetensors();
    test_dit_cache();
    test_rng_and_solver();
    test_rgb_resize();
    test_dit_row_conversions();
    test_metal_probe();
    test_terminal_zoom();
    test_video_encoder_single_frame_conv();
    test_video_vae_assembly();
    test_ffmpeg_stream();
    test_ffmpeg_video_tail();
    printf("ok: %d checks\n", tests_run);
    return 0;
}
