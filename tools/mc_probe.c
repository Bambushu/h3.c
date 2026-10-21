/* mc_probe CONTEXT.h3ctx CLIP.mp4 W H MODEL_DIR: encode the same tail both
 * ways (latent slice vs decoded video) and print per-step cosines. */
#include "h3_motion_context.h"
#include "h3_host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double cosine(const float *a, const float *b, size_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < n; i++) { ab += (double)a[i] * b[i]; aa += (double)a[i] * a[i]; bb += (double)b[i] * b[i]; }
    return ab / sqrt(aa * bb);
}
static double rms(const float *a, size_t n) {
    double s = 0; for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i]; return sqrt(s / (double)n);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: %s CONTEXT CLIP W H MODEL_DIR\n", argv[0]); return 2; }
    int w = atoi(argv[3]), h = atoi(argv[4]);
    int lw, lh; h3_latent_canvas(w, h, &lw, &lh);
    char error[512];
    char vae[1024], avae[1024];
    snprintf(vae, sizeof(vae), "%s/FL2VA/video_vae/source", argv[5]);
    snprintf(avae, sizeof(avae), "%s/FL2VA/audio_vae", argv[5]);
    h3_motion_context from_latent, from_video;
    if (!h3_motion_context_plan(argv[1], lh, lw, &from_latent, error, sizeof(error)) ||
        !h3_motion_context_encode(&from_latent, argv[1], w, h, NULL, NULL, NULL, NULL, NULL, NULL, NULL, error, sizeof(error))) {
        fprintf(stderr, "latent: %s\n", error); return 1; }
    if (!h3_motion_context_plan(argv[2], lh, lw, &from_video, error, sizeof(error)) ||
        !h3_motion_context_encode(&from_video, argv[2], w, h, vae, avae, "h3_shaders.metal", NULL, NULL, NULL, NULL, error, sizeof(error))) {
        fprintf(stderr, "video: %s\n", error); return 1; }
    printf("steps %d/%d  video elements %zu/%zu  audio %d/%d offset %+.1f/%+.1f\n",
           from_latent.steps, from_video.steps, from_latent.video_elements, from_video.video_elements,
           from_latent.audio_t, from_video.audio_t, from_latent.audio_offset, from_video.audio_offset);
    size_t per_step = from_latent.video_elements / (size_t)from_latent.steps;
    for (int s = 0; s < from_latent.steps; s++) {
        const float *a = from_latent.video_rows + (size_t)s * per_step;
        printf("video step %d: cos(latent, video) = %.3f  rms %.3f / %.3f", s, cosine(a, from_video.video_rows + (size_t)s * per_step, per_step), rms(a, per_step), rms(from_video.video_rows + (size_t)s * per_step, per_step));
        /* temporal shift check: this latent step against each video step */
        int best = -1; double best_cos = -2;
        for (int t = 0; t < from_video.steps; t++) {
            double c = cosine(a, from_video.video_rows + (size_t)t * per_step, per_step);
            if (c > best_cos) { best_cos = c; best = t; }
        }
        printf("  best video step %d (%.3f)\n", best, best_cos);
    }
    printf("audio rows: cos = %.3f  rms %.3f / %.3f\n",
           cosine(from_latent.audio_rows, from_video.audio_rows, from_latent.audio_elements),
           rms(from_latent.audio_rows, from_latent.audio_elements), rms(from_video.audio_rows, from_video.audio_elements));
    /* audio per time step (stereo-major rows: [2][t][32]) */
    int t_n = from_latent.audio_t;
    for (int t = 0; t < t_n; t += 6)
        printf("audio t %2d: cos %.3f\n", t, cosine(from_latent.audio_rows + (size_t)t * 32, from_video.audio_rows + (size_t)t * 32, 32));
    return 0;
}
