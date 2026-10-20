#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HEADS 56u
#define HEAD_DIM 128u

static uint32_t rng_state = 0x65859680u;

static float random_value(float scale) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return ((float)(rng_state & 0xffffu) / 32767.5f - 1.0f) * scale;
}

static uint16_t to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + rounding) >> 16);
}

static float from_bf16(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static double now_seconds(void) {
    struct timespec time;
    (void)clock_gettime(CLOCK_MONOTONIC, &time);
    return (double)time.tv_sec + (double)time.tv_nsec / 1e9;
}

static int read_payload(const char *path, uint16_t *values, size_t count) {
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    size_t read = fread(values, sizeof(*values), count, file);
    int extra = fgetc(file);
    fclose(file);
    return read == count && extra == EOF;
}

static int run_dense(h3_gpu *gpu, h3_gpu_tensor *output,
                     h3_gpu_tensor *query32, h3_gpu_tensor *key32,
                     h3_gpu_tensor *value32, h3_gpu_tensor *output32,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence) {
    uint32_t count = sequence * HEADS * HEAD_DIM;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    return h3_gpu_begin(gpu) &&
        h3_gpu_cast_bf16_to_f32(gpu, query32, query, count) &&
        h3_gpu_cast_bf16_to_f32(gpu, key32, key, count) &&
        h3_gpu_cast_bf16_to_f32(gpu, value32, value, count) &&
        h3_gpu_sdpa_f32(gpu, output32, query32, key32, value32,
                        sequence, HEADS, HEAD_DIM, scale) &&
        h3_gpu_cast_f32_to_bf16(gpu, output, output32, count) &&
        h3_gpu_submit(gpu);
}

static int run_sol(h3_gpu *gpu, h3_gpu_tensor *output,
                   const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                   const h3_gpu_tensor *value,
                   h3_gpu_tensor *query_centroids,
                   h3_gpu_tensor *key_centroids,
                   h3_gpu_tensor *value_sums,
                   h3_gpu_tensor *key_means,
                   h3_gpu_tensor *key_variances,
                   h3_gpu_tensor *thresholds,
                   uint32_t sequence, uint32_t sink_tokens, float tau,
                   int head_major_output) {
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    return h3_gpu_begin(gpu) && h3_gpu_sol_attention_bf16(
        gpu, output, query, key, value, query_centroids, key_centroids,
        value_sums, key_means, key_variances, thresholds, sequence, HEADS,
        HEAD_DIM, sink_tokens, scale, tau, head_major_output) &&
        h3_gpu_submit(gpu);
}

static int read_route_density(h3_gpu *gpu, h3_gpu_tensor *routes,
                              const h3_gpu_tensor *query_centroids,
                              const h3_gpu_tensor *key_centroids,
                              const h3_gpu_tensor *thresholds,
                              uint32_t sequence, uint32_t sink_tokens,
                              double *density) {
    uint32_t blocks = (sequence + 63u) / 64u;
    size_t count = (size_t)HEADS * blocks * blocks;
    int8_t *host = malloc(count);
    if (!host) return 0;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);
    int ok = h3_gpu_begin(gpu) &&
        h3_gpu_sol_attention_routes_bf16(
            gpu, routes, query_centroids, key_centroids, thresholds,
            sequence, HEADS, HEAD_DIM, sink_tokens, scale) &&
        h3_gpu_submit(gpu) && h3_gpu_tensor_read_i8(routes, host, count);
    size_t selected = 0;
    if (ok)
        for (size_t index = 0; index < count; index++)
            selected += host[index] != 0;
    free(host);
    if (ok) *density = (double)selected / (double)count;
    return ok;
}

static void fail_gpu(h3_gpu *gpu, const char *operation) {
    fprintf(stderr, "FAIL %s: %s\n", operation, h3_gpu_error(gpu));
}

int main(int argc, char **argv) {
    (void)setenv("H3_SOL_ATTN", "1", 0);
    const char *shaders = argc > 1 ? argv[1] : "h3_shaders.metal";
    uint32_t sequence = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 10) : 976u;
    unsigned iterations = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 3u;
    float tau = argc > 4 ? strtof(argv[4], NULL) : -100.0f;
    uint32_t sink_tokens = argc > 5 ?
        (uint32_t)strtoul(argv[5], NULL, 10) : 0u;
    const char *capture = argc > 6 && strcmp(argv[6], "-") ? argv[6] : NULL;
    int head_major_output = argc > 7 && strcmp(argv[7], "0");
    if (!sequence || !iterations || sequence > UINT32_MAX / HEADS / HEAD_DIM) {
        fputs("usage: h3_sol_attention_bench [shader] [tokens] [iterations] "
              "[tau] [sink-tokens] [capture-prefix|-] [head-major]\n", stderr);
        return 2;
    }

    char error[8192];
    h3_gpu *gpu = h3_gpu_create(shaders, error, sizeof(error));
    if (!gpu) {
        fprintf(stderr, "FAIL Sol-Attn Metal setup: %s\n", error);
        return 1;
    }
    size_t count = (size_t)sequence * HEADS * HEAD_DIM;
    uint32_t blocks = (sequence + 63u) / 64u;
    size_t summary_count = (size_t)HEADS * blocks * HEAD_DIM;
    size_t stats_count = (size_t)HEADS * HEAD_DIM;
    size_t threshold_count = (size_t)HEADS * blocks;
    size_t route_count = (size_t)HEADS * blocks * blocks;
    uint16_t *host = malloc(count * sizeof(*host));
    uint16_t *dense_values = malloc(count * sizeof(*dense_values));
    uint16_t *sol_values = malloc(count * sizeof(*sol_values));
    if (!host || !dense_values || !sol_values) {
        fputs("FAIL host allocation\n", stderr);
        return 1;
    }

    h3_gpu_tensor *query = NULL;
    h3_gpu_tensor *key = NULL;
    h3_gpu_tensor *value = NULL;
    const char *suffixes[] = {".q.bf16", ".k.bf16", ".v.bf16"};
    h3_gpu_tensor **inputs[] = {&query, &key, &value};
    for (unsigned input = 0; input < 3; input++) {
        if (capture) {
            size_t path_size = strlen(capture) + strlen(suffixes[input]) + 1;
            char *path = malloc(path_size);
            if (!path) return 1;
            (void)snprintf(path, path_size, "%s%s", capture, suffixes[input]);
            if (!read_payload(path, host, count)) {
                fprintf(stderr, "FAIL reading %s for %zu BF16 values\n",
                        path, count);
                free(path);
                return 1;
            }
            free(path);
        } else {
            float value_scale = input == 2 ? 1.0f : 0.25f;
            for (size_t index = 0; index < count; index++)
                host[index] = to_bf16(random_value(value_scale));
        }
        *inputs[input] = h3_gpu_tensor_from_bf16(gpu, host, count);
    }

#define NEW_BF16(name, elements) \
    h3_gpu_tensor *name = h3_gpu_tensor_new_bf16(gpu, (elements))
#define NEW_F32(name, elements) \
    h3_gpu_tensor *name = h3_gpu_tensor_new_f32(gpu, (elements))
    NEW_BF16(dense, count);
    NEW_BF16(sol, count);
    NEW_F32(query32, count);
    NEW_F32(key32, count);
    NEW_F32(value32, count);
    NEW_F32(output32, count);
    NEW_F32(query_centroids, summary_count);
    NEW_BF16(key_centroids, summary_count);
    NEW_BF16(value_sums, summary_count);
    NEW_F32(key_means, stats_count);
    NEW_F32(key_variances, stats_count);
    NEW_F32(thresholds, threshold_count);
    h3_gpu_tensor *routes = h3_gpu_tensor_new_i8(gpu, route_count);
#undef NEW_BF16
#undef NEW_F32
    h3_gpu_tensor *allocated[] = {
        query, key, value, dense, sol, query32, key32, value32, output32,
        query_centroids, key_centroids, value_sums, key_means, key_variances,
        thresholds, routes
    };
    for (size_t index = 0; index < sizeof(allocated) / sizeof(*allocated);
         index++) {
        if (!allocated[index]) {
            fputs("FAIL GPU allocation\n", stderr);
            return 1;
        }
    }

    if (!run_dense(gpu, dense, query32, key32, value32, output32,
                   query, key, value, sequence)) {
        fail_gpu(gpu, "dense warm-up");
        return 1;
    }
    if (!run_sol(gpu, sol, query, key, value, query_centroids, key_centroids,
                 value_sums, key_means, key_variances, thresholds, sequence,
                 sink_tokens, tau, head_major_output)) {
        fail_gpu(gpu, "Sol-Attn warm-up");
        return 1;
    }

    double dense_start = now_seconds();
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        if (!run_dense(gpu, dense, query32, key32, value32, output32,
                       query, key, value, sequence)) {
            fail_gpu(gpu, "dense benchmark");
            return 1;
        }
    }
    double dense_seconds = (now_seconds() - dense_start) / iterations;
    double sol_start = now_seconds();
    for (unsigned iteration = 0; iteration < iterations; iteration++) {
        if (!run_sol(gpu, sol, query, key, value, query_centroids,
                     key_centroids, value_sums, key_means, key_variances,
                     thresholds, sequence, sink_tokens, tau,
                     head_major_output)) {
            fail_gpu(gpu, "Sol-Attn benchmark");
            return 1;
        }
    }
    double sol_seconds = (now_seconds() - sol_start) / iterations;

    double route_density = 0.0;
    if (!read_route_density(gpu, routes, query_centroids, key_centroids,
                            thresholds, sequence, sink_tokens,
                            &route_density)) {
        fail_gpu(gpu, "Sol-Attn route density");
        return 1;
    }

    if (!h3_gpu_tensor_read_bf16(dense, dense_values, count) ||
        !h3_gpu_tensor_read_bf16(sol, sol_values, count)) {
        fputs("FAIL output readback\n", stderr);
        return 1;
    }
    double dot = 0.0;
    double dense_norm = 0.0;
    double sol_norm = 0.0;
    double absolute_error = 0.0;
    double max_error = 0.0;
    for (size_t index = 0; index < count; index++) {
        double a = from_bf16(dense_values[index]);
        size_t token = index / (HEADS * HEAD_DIM);
        size_t remainder = index % (HEADS * HEAD_DIM);
        size_t head = remainder / HEAD_DIM;
        size_t dimension = remainder % HEAD_DIM;
        size_t sol_index = head_major_output ?
            (head * sequence + token) * HEAD_DIM + dimension : index;
        double b = from_bf16(sol_values[sol_index]);
        double difference = fabs(a - b);
        dot += a * b;
        dense_norm += a * a;
        sol_norm += b * b;
        absolute_error += difference;
        if (difference > max_error) max_error = difference;
    }
    double cosine = dot / sqrt(dense_norm * sol_norm);
    printf("tokens=%u heads=%u dim=%u tau=%g sink=%u layout=%s source=%s\n",
           sequence, HEADS, HEAD_DIM, tau, sink_tokens,
           head_major_output ? "head-major" : "token-major",
           capture ? capture : "deterministic-random");
    printf("dense=%.6fs sol=%.6fs speedup=%.3fx routes=%.6f cosine=%.9f "
           "mae=%.9f max_error=%.9f\n",
           dense_seconds, sol_seconds, dense_seconds / sol_seconds,
           route_density, cosine,
           absolute_error / (double)count, max_error);

    int exact_control = sink_tokens >= sequence;
    int failed = !isfinite(cosine) || !isfinite(route_density) ||
        route_density <= 0.0 || route_density > 1.0 ||
        (exact_control && (cosine < 0.99999 ||
                           absolute_error / (double)count > 1e-5 ||
                           route_density < 0.999999));
    for (size_t index = 0; index < sizeof(allocated) / sizeof(*allocated);
         index++) h3_gpu_tensor_free(allocated[index]);
    h3_gpu_free(gpu);
    free(host);
    free(dense_values);
    free(sol_values);
    return failed;
}
