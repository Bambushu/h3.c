/* Diagnostic: apply a LoRA to every target of a BF16 checkpoint on the CPU,
 * exactly as the loader does, and compare against a checkpoint folded offline
 * by tools/fold_turbo_lora.py. Prints mismatching elements and the largest
 * absolute difference; exits 0 only when every target is byte-identical.
 *
 *   lora_check BASE_TRANSFORMER FOLDED_TRANSFORMER ADAPTER [SCALE] */
#include "h3_safetensors.h"
#include "h3_weights.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bf16_value(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    memcpy(&value, &wide, sizeof(value));
    return value;
}

static uint16_t *read_bf16(const h3_weight_store *store, const char *name,
                           size_t elements, char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    uint16_t *values = tensor ? malloc(elements * sizeof(*values)) : NULL;
    if (!values || !h3_st_read_data(header, tensor, values,
                                    elements * sizeof(*values),
                                    error, error_size)) {
        if (!tensor) snprintf(error, error_size, "%s is absent", name);
        free(values);
        return NULL;
    }
    return values;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "usage: %s BASE FOLDED ADAPTER [SCALE]\n", argv[0]);
        return 2;
    }
    float scale = argc == 5 ? strtof(argv[4], NULL) : 1.0f;
    char error[512] = {0};
    h3_lora *lora = h3_lora_open(argv[3], scale, error, sizeof(error));
    h3_weight_store *base = lora ?
        h3_weight_store_open_lora(argv[1], lora, error, sizeof(error)) : NULL;
    h3_weight_store *folded = base ?
        h3_weight_store_open(argv[2], error, sizeof(error)) : NULL;
    h3_st_header adapter = {0};
    if (!folded || !h3_st_read_header(argv[3], &adapter, error, sizeof(error))) {
        fprintf(stderr, "lora_check: %s\n", error);
        return 1;
    }
    printf("adapter key %016llx, %zu pairs\n",
           (unsigned long long)h3_lora_key(lora), h3_lora_pairs(lora));
    static const char suffix[] = ".lora_A.weight";
    size_t checked = 0, identical = 0;
    uint64_t mismatched = 0, total = 0;
    float worst = 0.0f;
    for (size_t index = 0; index < adapter.tensor_count; index++) {
        const char *key = adapter.tensors[index].name;
        size_t length = strlen(key);
        if (length <= sizeof(suffix) - 1 ||
            strcmp(key + length - (sizeof(suffix) - 1), suffix)) continue;
        char name[256];
        snprintf(name, sizeof(name), "%.*s.weight",
                 (int)(length - (sizeof(suffix) - 1)), key);
        const h3_st_tensor *tensor = h3_weight_find(base, name, NULL);
        uint64_t rows = tensor->shape[0], columns = tensor->shape[1];
        size_t elements = (size_t)(rows * columns);
        uint16_t *adapted = read_bf16(base, name, elements, error,
                                      sizeof(error));
        uint16_t *expected = adapted ?
            read_bf16(folded, name, elements, error, sizeof(error)) : NULL;
        if (!expected || h3_lora_apply(lora, name, adapted, rows, columns,
                                       error, sizeof(error)) != 1) {
            fprintf(stderr, "lora_check: %s: %s\n", name, error);
            return 1;
        }
        uint64_t differ = 0;
        for (size_t at = 0; at < elements; at++) {
            if (adapted[at] == expected[at]) continue;
            differ++;
            float gap = fabsf(bf16_value(adapted[at]) -
                              bf16_value(expected[at]));
            if (gap > worst) worst = gap;
        }
        checked++;
        identical += differ == 0;
        mismatched += differ;
        total += elements;
        if (differ)
            printf("%s: %llu of %zu elements differ\n", name,
                   (unsigned long long)differ, elements);
        free(adapted);
        free(expected);
    }
    printf("%zu targets, %zu byte-identical; %llu of %llu elements differ, "
           "max abs diff %g\n", checked, identical,
           (unsigned long long)mismatched, (unsigned long long)total,
           (double)worst);
    h3_st_free_header(&adapter);
    h3_weight_store_free(folded);
    h3_weight_store_free(base);
    h3_lora_free(lora);
    return checked && identical == checked ? 0 : 1;
}
