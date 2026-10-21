/* The flat setters must land on the h3_params field of the same name. The
 * wrapper is compiled in so the test can read the struct it fills. */
#include "../h3_flat.c"

#include <stdio.h>

static int failures;
#define CHECK(cond) do { if (!(cond)) { \
    fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
    failures++; } } while (0)

int main(void) {
    h3_flat_params *flat = h3_flat_params_new();
    h3_params defaults = H3_PARAMS_DEFAULT;
    CHECK(flat && !memcmp(&flat->params, &defaults, sizeof(defaults)));

    /* Each int field gets a distinct value and is read back directly. */
    int value = 1000;
    for (size_t i = 0; i < sizeof(int_fields) / sizeof(int_fields[0]); i++)
        CHECK(h3_flat_params_set_int(flat, int_fields[i].name, value++));
    const h3_params *p = &flat->params;
    CHECK(p->width == 1000 && p->height == 1001 && p->frames == 1002);
    CHECK(p->steps == 1003 && p->denoise_reuse == 1004);
    CHECK(p->dit_layers == 1005 && p->core_reuse == 1006);
    CHECK(p->token_reduction == 1007 && p->use_int8_row_fc2 == 1008);
    CHECK(p->use_reference_rope == 1009 && p->ssd_streaming == 1010);
    CHECK(p->render_width == 1011 && p->render_height == 1012);
    CHECK(p->use_slower_grouped_quantizer == 1022);
    CHECK(p->preview_denoise == 1023 && p->sampler == 1024);
    CHECK(p->scheduler == 1025);
    CHECK(h3_flat_params_set_int(flat, "reference_image_size", 1) &&
          p->reference_image_size == H3_REFERENCE_IMAGE_MAX);
    CHECK(!h3_flat_params_set_int(flat, "widht", 1));
    CHECK(!h3_flat_params_set_int(flat, "seed", 1));

    CHECK(h3_flat_params_set_u64(flat, "seed", UINT64_C(1) << 40) &&
          p->seed == (UINT64_C(1) << 40));
    CHECK(h3_flat_params_set_double(flat, "video_shift", 7.5) &&
          p->video_shift == 7.5);
    CHECK(!h3_flat_params_set_double(flat, "shift", 7.5));

    char scratch[32] = "/tmp/a.h3ctx";
    CHECK(h3_flat_params_set_string(flat, "save_context", scratch));
    scratch[0] = 'X';  /* the setter copied it */
    CHECK(!strcmp(flat->strings[FLAT_SAVE], "/tmp/a.h3ctx"));
    CHECK(h3_flat_params_set_string(flat, "save_context", ""));
    CHECK(flat->strings[FLAT_SAVE] == NULL);
    CHECK(!h3_flat_params_set_string(flat, "output", "x"));

    CHECK(h3_flat_params_add_reference(flat, H3_REFERENCE_IMAGE, "/a.png",
                                       NULL, 0));
    CHECK(h3_flat_params_add_reference(flat, H3_REFERENCE_VIDEO, "/b.mp4",
                                       "/b.wav", 1));
    CHECK(!h3_flat_params_add_reference(flat, H3_REFERENCE_AUDIO, "", NULL, 0));
    CHECK(flat->reference_count == 2);
    CHECK(flat->references[1].kind == H3_REFERENCE_VIDEO &&
          !strcmp(flat->references[1].path, "/b.mp4") &&
          !strcmp(flat->references[1].audio_path, "/b.wav") &&
          flat->references[1].include_embedded_audio == 1 &&
          flat->references[0].audio_path == NULL);
    h3_flat_params_clear_references(flat);
    CHECK(flat->reference_count == 0 && flat->references == NULL);
    h3_flat_params_free(flat);

    void *lock = h3_flat_lock_acquire();
    CHECK(lock != NULL);
    h3_flat_lock_release(lock);

    if (failures) return 1;
    printf("h3_flat: all checks passed\n");
    return 0;
}
