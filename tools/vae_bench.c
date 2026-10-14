/* Replay a video VAE decode from a latent saved with H3_DUMP_VIDEO_LATENT.
 *
 *   vae_bench <video_vae/source dir> <latent.bin> [out.rgbf32] [ref.rgbf32]
 *
 * Prints load/decode wall time and SHA-256 of the decoded F32 RGB and of the
 * RGB24 bytes H3 encodes, so decode paths (H3_VAE_DECODE_LEGACY=1,
 * H3_VAE_UNFUSED=1) can be checked for bit-identical output; with a reference
 * it prints max abs error and PSNR over the [0,1] RGB output, so precision
 * variants (H3_VAE_PRECISION) can be compared against the released F32 path
 * on identical input. VAE_BENCH_REPEAT=N decodes N times (the first decode
 * also builds pipelines and graphs) and checks every repeat matches. */
#include "h3_video_vae.h"

#include <CommonCrypto/CommonDigest.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *read_all(const char *path, size_t *size) {
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
    if (data) *size = (size_t)end;
    return data;
}

static void sha256_hex(const void *data, size_t bytes, char hex[65]) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const unsigned char *cursor = data;
    while (bytes) {
        CC_LONG part = bytes > 0x40000000u ? 0x40000000u : (CC_LONG)bytes;
        CC_SHA256_Update(&context, cursor, part);
        cursor += part;
        bytes -= part;
    }
    CC_SHA256_Final(digest, &context);
    for (int index = 0; index < CC_SHA256_DIGEST_LENGTH; index++)
        snprintf(hex + 2 * index, 3, "%02x", digest[index]);
}

/* h3.c's F32 -> RGB24 conversion. */
static uint8_t *to_rgb24(const float *rgb, size_t count) {
    uint8_t *output = malloc(count);
    if (!output) return NULL;
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
    return output;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s VAE_DIR LATENT [OUT] [REF]\n", argv[0]);
        return 2;
    }
    size_t size = 0;
    int32_t *blob = read_all(argv[2], &size);
    if (!blob || size < 12) {
        fprintf(stderr, "cannot read latent %s\n", argv[2]);
        return 1;
    }
    int t = blob[0], h = blob[1], w = blob[2];
    size_t count = (size_t)24 * (size_t)t * (size_t)h * (size_t)w;
    if (size != 12 + count * sizeof(float)) {
        fprintf(stderr, "latent size mismatch\n");
        return 1;
    }
    const float *latent = (const float *)(blob + 3);
    char error[512];
    double start = now();
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        argv[1], "h3_shaders.metal", h, w, NULL, NULL, error, sizeof(error));
    if (!decoder) {
        fprintf(stderr, "load: %s\n", error);
        return 1;
    }
    double loaded = now();
    const char *repeat_value = getenv("VAE_BENCH_REPEAT");
    int repeats = repeat_value ? atoi(repeat_value) : 1;
    if (repeats < 1) repeats = 1;
    printf("mode %s%s precision %s\n",
           h3_video_vae_pipelined() ? "pipelined" : "legacy",
           getenv("H3_VAE_UNFUSED") ? " (unfused)" : "",
           getenv("H3_VAE_PRECISION") ? getenv("H3_VAE_PRECISION") : "f32");
    h3_video_frames frames = {0};
    char first_hash[65] = {0};
    size_t values = 0;
    for (int run = 0; run < repeats; run++) {
        if (run) h3_video_frames_free(&frames);
        double started = now();
        if (!h3_video_vae_decoder_decode(decoder, latent, t, &frames, error,
                                         sizeof(error))) {
            fprintf(stderr, "decode: %s\n", error);
            return 1;
        }
        double decoded = now();
        values = (size_t)frames.frames * (size_t)frames.height *
                 (size_t)frames.width * 3;
        char f32_hash[65], rgb24_hash[65];
        sha256_hex(frames.rgb, values * sizeof(float), f32_hash);
        uint8_t *rgb24 = to_rgb24(frames.rgb, values);
        if (!rgb24) {
            fprintf(stderr, "out of memory\n");
            return 1;
        }
        sha256_hex(rgb24, values, rgb24_hash);
        free(rgb24);
        printf("latent %dx%dx%d -> %d frames %dx%d  load %.2fs  decode %.2fs\n",
               t, h, w, frames.frames, frames.width, frames.height,
               loaded - start, decoded - started);
        printf("sha256 rgbf32 %s  rgb24 %s\n", f32_hash, rgb24_hash);
        if (!run) memcpy(first_hash, f32_hash, sizeof(first_hash));
        else if (strcmp(first_hash, f32_hash)) {
            fprintf(stderr, "repeat %d decoded different output\n", run + 1);
            return 1;
        }
    }
    if (argc > 3 && argv[3][0] && strcmp(argv[3], "-") != 0) {
        FILE *out = fopen(argv[3], "wb");
        if (!out || fwrite(frames.rgb, sizeof(float), values, out) != values) {
            fprintf(stderr, "cannot write %s\n", argv[3]);
            return 1;
        }
        fclose(out);
    }
    if (argc > 4) {
        size_t ref_size = 0;
        float *ref = read_all(argv[4], &ref_size);
        if (!ref || ref_size != values * sizeof(float)) {
            fprintf(stderr, "reference size mismatch\n");
            return 1;
        }
        double max_error = 0.0, square = 0.0;
        for (size_t index = 0; index < values; index++) {
            double delta = (double)frames.rgb[index] - ref[index];
            if (fabs(delta) > max_error) max_error = fabs(delta);
            square += delta * delta;
        }
        double mse = square / (double)values;
        printf("vs ref: max_abs %.5f  rmse %.6f  psnr %.2f dB\n", max_error,
               sqrt(mse), mse > 0 ? 10.0 * log10(1.0 / mse) : INFINITY);
        free(ref);
    }
    free(frames.rgb);
    h3_video_vae_decoder_free(decoder);
    free(blob);
    return 0;
}
