#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    /* Counters and times cover this encode call; live/peak bytes describe
     * the encoder's shared Metal context. */
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef void (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

typedef struct h3_video_encoder h3_video_encoder;

/* Create one Metal context and load the released visual VAE encoder weights.
 * One encoder may encode any number of inputs sequentially; conv graphs and
 * scratch buffers are shared between calls. */
h3_video_encoder *h3_video_encoder_open(const char *weight_directory,
                                        const char *shader_source_path,
                                        char *error, size_t error_size);
/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved.
 * Single-frame inputs run in a reused scratch arena with one wait per tile and
 * execute every causal Conv3d as the equivalent Conv2d over its last temporal
 * tap. H3_VAE_ENCODER_CONV3D=1 keeps the released Conv3d kernels in that
 * pipeline (bit-identical to the released path); H3_DISABLE_VAE_ENCODER_ARENA=1
 * restores the released per-block allocation path, as multi-frame inputs use. */
int h3_video_encoder_encode(h3_video_encoder *encoder, const float *pixels,
                            int frames, int height, int width,
                            h3_video_encoder_progress progress,
                            void *progress_opaque, h3_video_latent *output,
                            char *error, size_t error_size);
void h3_video_encoder_free(h3_video_encoder *encoder);
void h3_video_latent_free(h3_video_latent *latent);

/* Copy temporal tap `tap` of an OIDHW [out,in,kernel,kernel,kernel] Conv3d
 * weight into an OIHW [out,in,kernel,kernel] Conv2d weight. A causal Conv3d
 * with kernel-1 zero front frames sees real data only in its last tap
 * (tap = kernel - 1) when the input has one frame. */
void h3_video_encoder_slice_tap_f32(const float *weight, uint32_t output_channels,
                                    uint32_t input_channels, uint32_t kernel,
                                    uint32_t tap, float *planar);

#endif
