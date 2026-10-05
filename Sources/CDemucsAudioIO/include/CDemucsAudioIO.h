#ifndef CDEMUCS_AUDIO_IO_H
#define CDEMUCS_AUDIO_IO_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct demucs_pcm_wav demucs_pcm_wav;
typedef struct {
    int sample_rate, channels, bits_per_sample, is_float;
    int64_t frames;
} demucs_pcm_wav_info;
// 1 = supported native-rate RIFF PCM, 0 = use the platform decoder,
// negative errno = I/O failure. The reader owns one open file until closed.
int demucs_open_pcm_wav(const char *path, int sample_rate,
    demucs_pcm_wav **reader, demucs_pcm_wav_info *info);
int demucs_read_pcm_wav(demucs_pcm_wav *reader, float *planes,
    int64_t capacity, ptrdiff_t plane_stride,
    int (*cancelled)(void *), void *context);
void demucs_close_pcm_wav(demucs_pcm_wav *reader);
// Samples remain owned by the caller for the entire synchronous write.
// dtype: 0 = float32, 1 = float16. Strides are signed element strides.
int demucs_write_wav(const char *path, const void *samples, int dtype,
    int64_t frames, int channels, ptrdiff_t frame_stride, ptrdiff_t channel_stride,
    int sample_rate, int float32_output, int (*cancelled)(void *), void *context);
#ifdef __cplusplus
}
#endif
#endif
