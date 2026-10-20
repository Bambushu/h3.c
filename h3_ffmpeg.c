#include "h3_ffmpeg.h"
#include "h3.h"

#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

static const char *ffmpeg_program(void) {
    const char *override = getenv("H3_FFMPEG");
    return override && *override ? override : "ffmpeg";
}

static const char *ffprobe_program(void) {
    const char *override = getenv("H3_FFPROBE");
    return override && *override ? override : "ffprobe";
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int make_parents(const char *path, char *error, size_t error_size) {
    char *copy = strdup(path);
    if (!copy) {
        fail(error, error_size, "out of memory resolving output directory");
        return 0;
    }
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
            fail(error, error_size, "cannot create output directory %s: %s",
                 copy, strerror(errno));
            free(copy);
            return 0;
        }
        *cursor = '/';
    }
    free(copy);
    return 1;
}

static int write_all(int descriptor, const uint8_t *data, size_t bytes,
                     char *error, size_t error_size) {
    while (bytes) {
        ssize_t written = write(descriptor, data,
                                bytes > (size_t)SSIZE_MAX ?
                                (size_t)SSIZE_MAX : bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            fail(error, error_size, "cannot stream RGB frames to FFmpeg: %s",
                 written < 0 ? strerror(errno) : "short write");
            return 0;
        }
        data += (size_t)written;
        bytes -= (size_t)written;
    }
    return 1;
}

int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width) *width = 0;
    if (height) *height = 0;
    if (!path || !*path || !width || !height) {
        fail(error, error_size, "invalid FFprobe visual-size arguments");
        return 0;
    }
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFprobe pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffprobe_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        fail(error, error_size, "cannot start FFprobe: %s", strerror(code));
        return 0;
    }
    char output[128];
    size_t received = 0;
    int overflow = 0;
    while (1) {
        char byte;
        ssize_t amount = read(stream[0], &byte, 1);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        if (received + 1 < sizeof(output)) output[received++] = byte;
        else overflow = 1;
    }
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFprobe: %s", strerror(errno));
        return 0;
    }
    output[received] = '\0';
    int parsed_width = 0, parsed_height = 0, consumed = 0;
    if (overflow || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        sscanf(output, "%dx%d%n", &parsed_width, &parsed_height, &consumed) != 2) {
        fail(error, error_size, "FFprobe could not inspect visual stream %s",
             path);
        return 0;
    }
    for (char *cursor = output + consumed; *cursor; cursor++) {
        if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' &&
            *cursor != '\n') {
            fail(error, error_size, "FFprobe returned an invalid visual size");
            return 0;
        }
    }
    if (parsed_width < 1 || parsed_height < 1) {
        fail(error, error_size, "visual stream has invalid dimensions %dx%d",
             parsed_width, parsed_height);
        return 0;
    }
    *width = parsed_width;
    *height = parsed_height;
    return 1;
}

int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1 ||
        (fit != H3_IMAGE_FIT_STRETCH && fit != H3_IMAGE_FIT_COVER)) {
        fail(error, error_size, "invalid FFmpeg image input arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3 ||
        area * 3 > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t bytes = area * 3;
    uint8_t *rgb = malloc(bytes);
    float *channel_major = malloc(bytes * sizeof(*channel_major));
    if (!rgb || !channel_major) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "out of memory decoding input image");
        return 0;
    }
    char filter[256];
    if (fit == H3_IMAGE_FIT_STRETCH) {
        snprintf(filter, sizeof(filter), "scale=%d:%d:flags=lanczos",
                 width, height);
    } else {
        snprintf(filter, sizeof(filter),
                 "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,"
                 "crop=%d:%d", width, height, width, height);
    }
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot create FFmpeg image pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-frames:v", "1", "-vf", filter,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < bytes) {
        ssize_t amount = read(stream[0], rgb + received, bytes - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received != bytes || trailing != 0) {
        free(rgb);
        free(channel_major);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            fail(error, error_size, "FFmpeg could not decode image %s (status %d)",
                 path, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        else
            fail(error, error_size,
                 "FFmpeg decoded %zu bytes for %dx%d image, expected %zu",
                 received + (trailing > 0 ? 1u : 0u), width, height, bytes);
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (size_t pixel = 0; pixel < area; pixel++) {
        channel_major[pixel] = (float)rgb[3 * pixel] * scale;
        channel_major[area + pixel] = (float)rgb[3 * pixel + 1] * scale;
        channel_major[2 * area + pixel] = (float)rgb[3 * pixel + 2] * scale;
    }
    free(rgb);
    *pixels = channel_major;
    return 1;
}

int h3_ffmpeg_read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 5 || (size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "invalid FFmpeg video input arguments");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3) {
        fail(error, error_size, "decoded video frame size overflows");
        return 0;
    }
    size_t frame_bytes = area * 3;
    if ((size_t)max_frames > SIZE_MAX / frame_bytes) {
        fail(error, error_size, "decoded video size overflows");
        return 0;
    }
    size_t capacity = (size_t)max_frames * frame_bytes;
    uint8_t *rgb = malloc(capacity);
    if (!rgb) {
        fail(error, error_size, "out of memory decoding input video");
        return 0;
    }
    char filter[256], frame_limit[32];
    snprintf(filter, sizeof(filter),
             "fps=24,scale=%d:%d:flags=lanczos,setsar=1", width, height);
    snprintf(frame_limit, sizeof(frame_limit), "%d", max_frames);
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        fail(error, error_size, "cannot create FFmpeg video pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:v:0", "-an", "-vf", filter,
        "-frames:v", frame_limit, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], rgb + received, capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 || trailing != 0 ||
        received % frame_bytes) {
        free(rgb);
        fail(error, error_size, "FFmpeg could not decode bounded video %s",
             path);
        return 0;
    }
    int frame_count = (int)(received / frame_bytes);
    if (frame_count < 5) {
        free(rgb);
        fail(error, error_size,
             "reference videos require at least 5 decoded frames");
        return 0;
    }
    while (frame_count % 17 != 5) frame_count--;
    if ((size_t)frame_count > SIZE_MAX / 3 / area ||
        (size_t)frame_count * 3 * area > SIZE_MAX / sizeof(float)) {
        free(rgb);
        fail(error, error_size, "converted video size overflows");
        return 0;
    }
    size_t values = (size_t)frame_count * 3 * area;
    float *channel_major = malloc(values * sizeof(*channel_major));
    if (!channel_major) {
        free(rgb);
        fail(error, error_size, "out of memory converting input video");
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    for (int time = 0; time < frame_count; time++)
        for (size_t pixel = 0; pixel < area; pixel++)
            for (size_t channel = 0; channel < 3; channel++) {
                size_t source = ((size_t)time * area + pixel) * 3 + channel;
                size_t destination = (channel * (size_t)frame_count +
                                      (size_t)time) * area + pixel;
                channel_major[destination] = (float)rgb[source] * scale;
            }
    free(rgb);
    *pixels = channel_major;
    *frames = frame_count;
    return 1;
}

int h3_ffmpeg_read_audio_f32(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size) {
    enum { AUDIO_CHANNELS = 2, MIN_SAMPLES = 64000 };
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < MIN_SAMPLES ||
        max_samples > H3_REFERENCE_AUDIO_MAX_SAMPLES ||
        (truncate_at_limit != 0 && truncate_at_limit != 1)) {
        fail(error, error_size, "invalid FFmpeg audio input arguments");
        return 0;
    }
    size_t elements = (size_t)max_samples * AUDIO_CHANNELS;
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded audio size overflows");
        return 0;
    }
    float *interleaved = malloc(elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory decoding reference audio");
        return 0;
    }
    char duration[64];
    double seconds = (double)max_samples / (double)H3_REFERENCE_AUDIO_RATE;
    if (!truncate_at_limit)
        seconds += 1.0 / (double)H3_REFERENCE_AUDIO_RATE;
    snprintf(duration, sizeof(duration), "%.9f", seconds);
    int stream[2];
    if (pipe(stream) != 0) {
        free(interleaved);
        fail(error, error_size, "cannot create FFmpeg audio pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "32000",
        "-t", duration, "-f", "f32le", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(interleaved);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t capacity = elements * sizeof(*interleaved);
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], (uint8_t *)interleaved + received,
                              capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(interleaved);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    size_t frame_bytes = AUDIO_CHANNELS * sizeof(float);
    if (!truncate_at_limit && trailing > 0) {
        free(interleaved);
        fail(error, error_size,
             "reference audio exceeds the %d second total limit",
             H3_REFERENCE_AUDIO_MAX_SECONDS);
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received % frame_bytes) {
        free(interleaved);
        fail(error, error_size,
             "FFmpeg could not decode a stereo soundtrack from %s", path);
        return 0;
    }
    int sample_count = (int)(received / frame_bytes);
    if (sample_count < MIN_SAMPLES) {
        free(interleaved);
        fail(error, error_size,
             "reference audio requires at least 2 seconds at 32 kHz");
        return 0;
    }
    size_t output_elements = (size_t)sample_count * AUDIO_CHANNELS;
    float *channel_major = malloc(output_elements * sizeof(*channel_major));
    if (!channel_major) {
        free(interleaved);
        fail(error, error_size, "out of memory converting reference audio");
        return 0;
    }
    for (int sample = 0; sample < sample_count; sample++)
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            channel_major[(size_t)channel * (size_t)sample_count +
                          (size_t)sample] =
                interleaved[(size_t)sample * AUDIO_CHANNELS +
                            (size_t)channel];
    free(interleaved);
    *pcm = channel_major;
    *samples = sample_count;
    return 1;
}

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
                          int frame_count, int width, int height, int fps,
                          char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!path || !*path || !frames || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2) {
        fail(error, error_size, "invalid FFmpeg RGB output arguments");
        return 0;
    }
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / 3 ||
        (size_t)frame_count > SIZE_MAX / (pixels * 3)) {
        fail(error, error_size, "RGB video size overflows");
        return 0;
    }
    if (!make_parents(path, error, error_size)) return 0;
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFmpeg pipe: %s", strerror(errno));
        return 0;
    }
    char size[64], rate[32];
    snprintf(size, sizeof(size), "%dx%d", width, height);
    snprintf(rate, sizeof(rate), "%d", fps);
    char *arguments[] = {
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", size, "-framerate", rate,
        "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(&actions, stream[0], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[0]);
    if (code) {
        close(stream[1]);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    struct sigaction ignore, previous;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    sigaction(SIGPIPE, &ignore, &previous);
    int ok = write_all(stream[1], frames,
                       (size_t)frame_count * pixels * 3,
                       error, error_size);
    close(stream[1]);
    sigaction(SIGPIPE, &previous, NULL);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        if (ok) fail(error, error_size, "cannot wait for FFmpeg: %s",
                     strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (ok) fail(error, error_size, "FFmpeg exited with status %d",
                     WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return ok;
}

typedef struct {
    int descriptor;
    const uint8_t *data;
    size_t bytes;
    int error;
} stream_writer;

static void *stream_thread(void *opaque) {
    stream_writer *writer = opaque;
    const uint8_t *data = writer->data;
    size_t remaining = writer->bytes;
    while (remaining) {
        size_t request = remaining > (size_t)SSIZE_MAX ?
                         (size_t)SSIZE_MAX : remaining;
        ssize_t written = write(writer->descriptor, data, request);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) {
            writer->error = written < 0 ? errno : EIO;
            break;
        }
        data += (size_t)written;
        remaining -= (size_t)written;
    }
    close(writer->descriptor);
    writer->descriptor = -1;
    return NULL;
}

static int close_action(posix_spawn_file_actions_t *actions, int descriptor,
                        int keep_a, int keep_b) {
    return descriptor == keep_a || descriptor == keep_b ? 0 :
           posix_spawn_file_actions_addclose(actions, descriptor);
}

static int validate_av_arguments(const char *path, int frame_count,
                                 int width, int height, int fps,
                                 const float *pcm, int samples, int channels,
                                 int sample_rate, char *error,
                                 size_t error_size) {
    if (!path || !*path || frame_count < 1 || width < 2 ||
        height < 2 || fps < 1 || width % 2 || height % 2 || !pcm ||
        samples < 1 || channels < 1 || sample_rate < 1) {
        fail(error, error_size, "invalid FFmpeg A/V output arguments");
        return 0;
    }
    size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / 3 ||
        (size_t)frame_count > SIZE_MAX / (pixels * 3) ||
        (size_t)samples > SIZE_MAX / (size_t)channels ||
        (size_t)samples * (size_t)channels > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "A/V stream size overflows");
        return 0;
    }
    return 1;
}

static float *interleave_pcm(const float *pcm, int samples, int channels,
                             char *error, size_t error_size) {
    size_t pcm_elements = (size_t)samples * (size_t)channels;
    float *interleaved = malloc(pcm_elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory interleaving generated PCM");
        return NULL;
    }
    for (int sample = 0; sample < samples; sample++)
        for (int channel = 0; channel < channels; channel++)
            interleaved[(size_t)sample * (size_t)channels + (size_t)channel] =
                pcm[(size_t)channel * (size_t)samples + (size_t)sample];
    return interleaved;
}

/* Start the H.264 + AAC encoder reading rgb24 on stdin and f32le PCM on a
 * second pipe; returns the child and the two write ends, or -1. */
static pid_t spawn_av_encoder(const char *path, int width, int height,
                              int fps, int channels, int sample_rate,
                              int *video_write, int *audio_write,
                              char *error, size_t error_size) {
    int video_pipe[2] = {-1, -1}, audio_pipe[2] = {-1, -1};
    if (pipe(video_pipe) != 0 || pipe(audio_pipe) != 0) {
        fail(error, error_size, "cannot create FFmpeg A/V pipes: %s",
             strerror(errno));
        if (video_pipe[0] >= 0) close(video_pipe[0]);
        if (video_pipe[1] >= 0) close(video_pipe[1]);
        if (audio_pipe[0] >= 0) close(audio_pipe[0]);
        if (audio_pipe[1] >= 0) close(audio_pipe[1]);
        return -1;
    }
    int audio_target = video_pipe[0];
    if (video_pipe[1] > audio_target) audio_target = video_pipe[1];
    if (audio_pipe[0] > audio_target) audio_target = audio_pipe[0];
    if (audio_pipe[1] > audio_target) audio_target = audio_pipe[1];
    audio_target++;
    char size[64], rate[32], audio_rate[32], audio_channels[32], audio_input[32];
    snprintf(size, sizeof(size), "%dx%d", width, height);
    snprintf(rate, sizeof(rate), "%d", fps);
    snprintf(audio_rate, sizeof(audio_rate), "%d", sample_rate);
    snprintf(audio_channels, sizeof(audio_channels), "%d", channels);
    snprintf(audio_input, sizeof(audio_input), "pipe:%d", audio_target);
    char *arguments[] = {
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", size, "-framerate", rate,
        "-i", "pipe:0",
        "-f", "f32le", "-ar", audio_rate, "-ac", audio_channels,
        "-i", audio_input,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart", (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, video_pipe[0], STDIN_FILENO);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, audio_pipe[0], audio_target);
    int descriptors[] = {video_pipe[0], video_pipe[1],
                         audio_pipe[0], audio_pipe[1]};
    for (size_t index = 0; !code && index < sizeof(descriptors) / sizeof(*descriptors);
         index++)
        code = close_action(&actions, descriptors[index], STDIN_FILENO,
                            audio_target);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(video_pipe[0]);
    close(audio_pipe[0]);
    if (code) {
        close(video_pipe[1]);
        close(audio_pipe[1]);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return -1;
    }
    *video_write = video_pipe[1];
    *audio_write = audio_pipe[1];
    return child;
}

static int wait_encoder(pid_t child, char *error, size_t error_size) {
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(error, error_size, "FFmpeg exited with status %d",
             WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return 1;
}

int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
                                 int frame_count, int width, int height,
                                 int fps, const float *pcm, int samples,
                                 int channels, int sample_rate,
                                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!frames) {
        fail(error, error_size, "invalid FFmpeg A/V output arguments");
        return 0;
    }
    if (!validate_av_arguments(path, frame_count, width, height, fps, pcm,
                               samples, channels, sample_rate, error,
                               error_size)) return 0;
    size_t pixels = (size_t)width * (size_t)height;
    if (!make_parents(path, error, error_size)) return 0;
    size_t pcm_elements = (size_t)samples * (size_t)channels;
    float *interleaved = interleave_pcm(pcm, samples, channels, error,
                                        error_size);
    if (!interleaved) return 0;
    int video_write = -1, audio_write = -1;
    pid_t child = spawn_av_encoder(path, width, height, fps, channels,
                                   sample_rate, &video_write, &audio_write,
                                   error, error_size);
    if (child < 0) {
        free(interleaved);
        return 0;
    }

    struct sigaction ignore, previous;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    sigaction(SIGPIPE, &ignore, &previous);
    stream_writer video = {
        video_write, frames, (size_t)frame_count * pixels * 3, 0
    };
    stream_writer audio = {
        audio_write, (const uint8_t *)interleaved,
        pcm_elements * sizeof(*interleaved), 0
    };
    pthread_t video_thread, audio_thread;
    int video_code = pthread_create(&video_thread, NULL, stream_thread, &video);
    int audio_code = pthread_create(&audio_thread, NULL, stream_thread, &audio);
    if (video_code) {
        close(video.descriptor);
        video.descriptor = -1;
    }
    if (audio_code) {
        close(audio.descriptor);
        audio.descriptor = -1;
    }
    if (!video_code) pthread_join(video_thread, NULL);
    if (!audio_code) pthread_join(audio_thread, NULL);
    sigaction(SIGPIPE, &previous, NULL);
    free(interleaved);

    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (video_code || audio_code) {
        fail(error, error_size, "cannot start FFmpeg stream thread: %s",
             strerror(video_code ? video_code : audio_code));
        return 0;
    }
    if (video.error || audio.error) {
        fail(error, error_size, "cannot stream generated A/V to FFmpeg: %s",
             strerror(video.error ? video.error : audio.error));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        fail(error, error_size, "FFmpeg exited with status %d",
             WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return 0;
    }
    return 1;
}

struct h3_ffmpeg_av_stream {
    pid_t child;
    char *path;
    char *temp_path;
    const uint8_t *frames;
    size_t frame_bytes;
    int frame_count;
    float *interleaved;
    stream_writer audio;
    int video_descriptor;
    int video_error;
    int ready;                  /* frames published by the caller */
    int sent;                   /* frames written to FFmpeg */
    int aborting;
    int video_started;
    int audio_started;
    pthread_t video_thread;
    pthread_t audio_thread;
    pthread_mutex_t lock;
    pthread_cond_t changed;
    struct sigaction previous_sigpipe;
};

/* Write each run of published frames as soon as it appears; the bytes and
 * their order are exactly those of the whole-buffer writer. */
static void *video_feed_thread(void *opaque) {
    h3_ffmpeg_av_stream *stream = opaque;
    pthread_mutex_lock(&stream->lock);
    while (stream->sent < stream->frame_count) {
        while (stream->ready == stream->sent && !stream->aborting)
            pthread_cond_wait(&stream->changed, &stream->lock);
        if (stream->aborting) break;
        int first = stream->sent, last = stream->ready;
        pthread_mutex_unlock(&stream->lock);
        stream_writer run = {
            stream->video_descriptor,
            stream->frames + (size_t)first * stream->frame_bytes,
            (size_t)(last - first) * stream->frame_bytes, 0
        };
        const uint8_t *data = run.data;
        size_t remaining = run.bytes;
        while (remaining) {
            size_t request = remaining > (size_t)SSIZE_MAX ?
                             (size_t)SSIZE_MAX : remaining;
            ssize_t written = write(run.descriptor, data, request);
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0) {
                run.error = written < 0 ? errno : EIO;
                break;
            }
            data += (size_t)written;
            remaining -= (size_t)written;
        }
        pthread_mutex_lock(&stream->lock);
        if (run.error) {
            stream->video_error = run.error;
            break;
        }
        stream->sent = last;
    }
    pthread_mutex_unlock(&stream->lock);
    close(stream->video_descriptor);
    stream->video_descriptor = -1;
    return NULL;
}

/* <directory>/.h3-partial-<pid>-<name>: same directory (atomic rename) and
 * same extension (FFmpeg picks the container from it). */
static char *partial_path(const char *path) {
    const char *slash = strrchr(path, '/');
    size_t directory = slash ? (size_t)(slash - path) + 1 : 0;
    size_t bytes = strlen(path) + 64;
    char *temp = malloc(bytes);
    if (!temp) return NULL;
    snprintf(temp, bytes, "%.*s.h3-partial-%ld-%s", (int)directory, path,
             (long)getpid(), path + directory);
    return temp;
}

static void stream_free(h3_ffmpeg_av_stream *stream) {
    pthread_mutex_destroy(&stream->lock);
    pthread_cond_destroy(&stream->changed);
    free(stream->interleaved);
    free(stream->path);
    free(stream->temp_path);
    free(stream);
}

static void stream_join(h3_ffmpeg_av_stream *stream) {
    if (stream->video_started) pthread_join(stream->video_thread, NULL);
    else if (stream->video_descriptor >= 0) close(stream->video_descriptor);
    if (stream->audio_started) pthread_join(stream->audio_thread, NULL);
    else if (stream->audio.descriptor >= 0) close(stream->audio.descriptor);
    stream->video_started = stream->audio_started = 0;
    sigaction(SIGPIPE, &stream->previous_sigpipe, NULL);
}

h3_ffmpeg_av_stream *h3_ffmpeg_av_stream_open(
        const char *path, const uint8_t *frames, int frame_count, int width,
        int height, int fps, const float *pcm, int samples, int channels,
        int sample_rate, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!frames) {
        fail(error, error_size, "invalid FFmpeg A/V output arguments");
        return NULL;
    }
    if (!validate_av_arguments(path, frame_count, width, height, fps, pcm,
                               samples, channels, sample_rate, error,
                               error_size) ||
        !make_parents(path, error, error_size)) return NULL;
    h3_ffmpeg_av_stream *stream = calloc(1, sizeof(*stream));
    if (!stream) {
        fail(error, error_size, "out of memory starting FFmpeg");
        return NULL;
    }
    pthread_mutex_init(&stream->lock, NULL);
    pthread_cond_init(&stream->changed, NULL);
    stream->child = -1;
    stream->video_descriptor = -1;
    stream->audio.descriptor = -1;
    stream->frames = frames;
    stream->frame_bytes = (size_t)width * (size_t)height * 3;
    stream->frame_count = frame_count;
    stream->path = strdup(path);
    stream->temp_path = partial_path(path);
    if (!stream->path || !stream->temp_path) {
        fail(error, error_size, "out of memory starting FFmpeg");
        stream_free(stream);
        return NULL;
    }
    stream->interleaved = interleave_pcm(pcm, samples, channels, error,
                                         error_size);
    if (!stream->interleaved) {
        stream_free(stream);
        return NULL;
    }
    stream->child = spawn_av_encoder(stream->temp_path, width, height, fps,
                                     channels, sample_rate,
                                     &stream->video_descriptor,
                                     &stream->audio.descriptor,
                                     error, error_size);
    if (stream->child < 0) {
        stream_free(stream);
        return NULL;
    }
    struct sigaction ignore;
    memset(&ignore, 0, sizeof(ignore));
    ignore.sa_handler = SIG_IGN;
    sigemptyset(&ignore.sa_mask);
    sigaction(SIGPIPE, &ignore, &stream->previous_sigpipe);
    stream->audio.data = (const uint8_t *)stream->interleaved;
    stream->audio.bytes = (size_t)samples * (size_t)channels *
                          sizeof(*stream->interleaved);
    int video_code = pthread_create(&stream->video_thread, NULL,
                                    video_feed_thread, stream);
    stream->video_started = !video_code;
    int audio_code = pthread_create(&stream->audio_thread, NULL,
                                    stream_thread, &stream->audio);
    stream->audio_started = !audio_code;
    if (video_code || audio_code) {
        fail(error, error_size, "cannot start FFmpeg stream thread: %s",
             strerror(video_code ? video_code : audio_code));
        h3_ffmpeg_av_stream_abort(stream);
        return NULL;
    }
    return stream;
}

void h3_ffmpeg_av_stream_ready(h3_ffmpeg_av_stream *stream, int frames) {
    if (!stream) return;
    pthread_mutex_lock(&stream->lock);
    if (frames > stream->ready && frames <= stream->frame_count) {
        stream->ready = frames;
        pthread_cond_signal(&stream->changed);
    }
    pthread_mutex_unlock(&stream->lock);
}

int h3_ffmpeg_av_stream_finish(h3_ffmpeg_av_stream *stream, char *error,
                               size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!stream) return 0;
    pthread_mutex_lock(&stream->lock);
    int complete = stream->ready == stream->frame_count;
    pthread_mutex_unlock(&stream->lock);
    if (!complete) {
        fail(error, error_size, "FFmpeg stream finished before every frame");
        h3_ffmpeg_av_stream_abort(stream);
        return 0;
    }
    stream_join(stream);
    int ok = wait_encoder(stream->child, error, error_size);
    stream->child = -1;
    if (ok && (stream->video_error || stream->audio.error)) {
        fail(error, error_size, "cannot stream generated A/V to FFmpeg: %s",
             strerror(stream->video_error ? stream->video_error :
                      stream->audio.error));
        ok = 0;
    }
    if (ok && rename(stream->temp_path, stream->path) != 0) {
        fail(error, error_size, "cannot move FFmpeg output to %s: %s",
             stream->path, strerror(errno));
        ok = 0;
    }
    if (!ok) unlink(stream->temp_path);
    stream_free(stream);
    return ok;
}

void h3_ffmpeg_av_stream_abort(h3_ffmpeg_av_stream *stream) {
    if (!stream) return;
    if (stream->child > 0) kill(stream->child, SIGKILL);
    pthread_mutex_lock(&stream->lock);
    stream->aborting = 1;
    pthread_cond_signal(&stream->changed);
    pthread_mutex_unlock(&stream->lock);
    stream_join(stream);
    if (stream->child > 0) {
        int status = 0;
        while (waitpid(stream->child, &status, 0) < 0 && errno == EINTR) {}
    }
    unlink(stream->temp_path);
    stream_free(stream);
}
