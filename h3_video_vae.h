#ifndef H3_VIDEO_VAE_H
#define H3_VIDEO_VAE_H

#include "h3_gpu.h"

#include <stddef.h>

typedef struct {
    int frames;
    int height;
    int width;
    /* Frame-major, row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_video_frames;

typedef void (*h3_video_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);

typedef struct h3_video_vae_decoder h3_video_vae_decoder;

/* Resident tiled decoder used by live denoising previews. It decodes one
 * representative middle frame per preview call and can then produce the final
 * complete video without loading the 9.7 GiB weight set again. */
h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size);
int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size);
int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size);

/* Receives finished output frames in display order: count frames starting at
 * global frame first, frame-major [height, width, 3] F32 in [0,1]. rgb is only
 * valid during the call. Return 0 to stop the decode. */
typedef int (*h3_video_frame_sink)(const float *rgb, int first, int count,
                                   int width, int height, void *opaque);
/* h3_video_vae_decoder_decode, delivering each temporal chunk's final frames
 * as soon as it is stitched and blended instead of one buffer at the end, so
 * conversion and encoding can overlap the rest of the decode. The frames are
 * bit-identical to h3_video_vae_decoder_decode. */
int h3_video_vae_decoder_decode_stream(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frame_sink sink, void *opaque,
                        char *error, size_t error_size);
void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder);
/* 1 unless H3_VAE_DECODE_LEGACY=1 selects the released resident decode (one
 * blocking submit per tile, unfused BF16 epilogues, whole-video hand-off). */
int h3_video_vae_pipelined(void);
/* Test hook: the pipelined host assembly (per-tile stitching, streamed chunk
 * delivery) must reproduce the legacy stitched frames bit for bit on
 * synthetic decoder output. Returns 1 on success. */
int h3_video_vae_assembly_selftest(char *error, size_t error_size);

/* Decode aligned H3 temporal chunks, using the released overlap/blend rules in
 * time and 256-pixel overlapping tiles in space. The two-token mode remains
 * available only for the diagnostic MLX fixture. */
int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_frames_free(h3_video_frames *frames);

#endif
