/* A/B the single-frame visual VAE encoder paths on real anchor images.
 *
 *   h3_vae_encoder_bench VAE_DIR FIRST [--last LAST] [--width W]
 *                        [--height H] [--passes N]
 *
 * VAE_DIR is a checkpoint's FL2VA/video_vae/source directory. Images are
 * fitted exactly as h3 does (first frame stretched, last frame aspect-cover);
 * the default canvas is 768x1344. Each pass opens one encoder per path, the
 * released Conv3d path (H3_VAE_ENCODER_CONV3D=1) and then the default
 * single-frame Conv2d path, and encodes every image with it. The tool prints
 * per-path open/encode wall time and GPU counters, then the latent max abs and
 * relative L2 difference between the paths. With PASSES > 1 it also prints
 * the run-to-run difference of each path as a noise floor. */
#include "h3_ffmpeg.h"
#include "h3_video_encoder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { PATHS = 2, MAX_IMAGES = 2 };

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void compare(const char *label, const h3_video_latent *a,
                    const h3_video_latent *b) {
    if (a->time != b->time || a->height != b->height || a->width != b->width) {
        printf("%s: shape mismatch\n", label);
        return;
    }
    size_t count = (size_t)24 * (size_t)a->time * (size_t)a->height *
                   (size_t)a->width;
    double maximum = 0.0, square_error = 0.0, square_value = 0.0;
    size_t differing = 0;
    for (size_t index = 0; index < count; index++) {
        double delta = (double)a->values[index] - b->values[index];
        if (delta != 0.0) differing++;
        if (fabs(delta) > maximum) maximum = fabs(delta);
        square_error += delta * delta;
        square_value += (double)a->values[index] * a->values[index];
    }
    printf("%s: max abs %.7g, relative L2 %.7g, %zu/%zu values differ%s\n",
           label, maximum,
           sqrt(square_error / (square_value > 1e-24 ? square_value : 1e-24)),
           differing, count, differing ? "" : " (bit-identical)");
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s VAE_DIR FIRST [--last LAST] [--width W] "
                "[--height H] [--passes N]\n", argv[0]);
        return 2;
    }
    const char *images[MAX_IMAGES] = {argv[2], NULL};
    int image_count = 1, width = 768, height = 1344, passes = 1;
    for (int index = 3; index + 1 < argc; index += 2) {
        if (!strcmp(argv[index], "--last")) {
            images[1] = argv[index + 1];
            image_count = 2;
        } else if (!strcmp(argv[index], "--width")) {
            width = atoi(argv[index + 1]);
        } else if (!strcmp(argv[index], "--height")) {
            height = atoi(argv[index + 1]);
        } else if (!strcmp(argv[index], "--passes")) {
            passes = atoi(argv[index + 1]);
        } else {
            fprintf(stderr, "unknown option %s\n", argv[index]);
            return 2;
        }
    }
    if ((argc - 3) % 2 || width < 32 || height < 32 || width % 16 ||
        height % 16 || passes < 1) {
        fprintf(stderr, "invalid arguments, canvas or pass count\n");
        return 2;
    }
    char error[512];
    float *pixels[MAX_IMAGES] = {0};
    for (int image = 0; image < image_count; image++)
        if (!h3_ffmpeg_read_image_f32(
                images[image], width, height,
                image ? H3_IMAGE_FIT_COVER : H3_IMAGE_FIT_STRETCH,
                &pixels[image], error, sizeof(error))) {
            fprintf(stderr, "%s\n", error);
            return 1;
        }
    static const char *names[PATHS] = {"conv3d", "conv2d"};
    static const char *values[PATHS] = {"1", "0"};
    h3_video_latent first[PATHS][MAX_IMAGES];
    h3_video_latent latest[PATHS][MAX_IMAGES];
    memset(first, 0, sizeof(first));
    memset(latest, 0, sizeof(latest));
    for (int pass = 0; pass < passes; pass++)
        for (int path = 0; path < PATHS; path++) {
            setenv("H3_VAE_ENCODER_CONV3D", values[path], 1);
            double start = now();
            h3_video_encoder *encoder = h3_video_encoder_open(
                argv[1], "h3_shaders.metal", error, sizeof(error));
            if (!encoder) {
                fprintf(stderr, "%s\n", error);
                return 1;
            }
            double opened = now();
            printf("pass %d %s: open %.3fs", pass + 1, names[path],
                   opened - start);
            for (int image = 0; image < image_count; image++) {
                h3_video_latent *slot = pass ? &latest[path][image] :
                                               &first[path][image];
                h3_video_latent_free(slot);
                double encode_start = now();
                if (!h3_video_encoder_encode(encoder, pixels[image], 1, height,
                                             width, NULL, NULL, slot, error,
                                             sizeof(error))) {
                    fprintf(stderr, "\n%s\n", error);
                    return 1;
                }
                printf(", image %d %.3fs (wait %.3fs, %llu submissions, "
                       "%llu convs)", image + 1, now() - encode_start,
                       slot->gpu_stats.command_wait_seconds,
                       (unsigned long long)slot->gpu_stats.submissions,
                       (unsigned long long)slot->gpu_stats.mps_conv_dispatches);
            }
            h3_video_encoder_free(encoder);
            printf(", total %.3fs\n", now() - start);
        }
    for (int image = 0; image < image_count; image++) {
        char label[96];
        int last = passes > 1;
        snprintf(label, sizeof(label), "image %d conv2d vs conv3d", image + 1);
        compare(label, last ? &latest[0][image] : &first[0][image],
                last ? &latest[1][image] : &first[1][image]);
        if (!last) continue;
        for (int path = 0; path < PATHS; path++) {
            snprintf(label, sizeof(label), "image %d %s pass 1 vs pass %d",
                     image + 1, names[path], passes);
            compare(label, &first[path][image], &latest[path][image]);
        }
    }
    for (int path = 0; path < PATHS; path++)
        for (int image = 0; image < image_count; image++) {
            h3_video_latent_free(&first[path][image]);
            h3_video_latent_free(&latest[path][image]);
        }
    for (int image = 0; image < image_count; image++) free(pixels[image]);
    return 0;
}
