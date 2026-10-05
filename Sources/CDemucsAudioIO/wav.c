// Strided, bounded WAV export adapted from mlx-audio-io.
// Copyright (c) 2025 ssmall256. MIT licensed; see LICENSE.
#include "CDemucsAudioIO.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

static int put(FILE *f, const void *p, size_t n) {
    return fwrite(p, 1, n, f) == n ? 0 : (errno ? errno : EIO);
}
static void u16(uint8_t *p, uint16_t x) { p[0] = x; p[1] = x >> 8; }
static void u32(uint8_t *p, uint32_t x) {
    p[0] = x; p[1] = x >> 8; p[2] = x >> 16; p[3] = x >> 24;
}
static float sample(const void *p, int dtype, ptrdiff_t i) {
    return dtype == 0 ? ((const float *)p)[i] : (float)((const _Float16 *)p)[i];
}
static int16_t pcm16(float x) {
    if (isnan(x)) return 0;
    if (x >= 1.0f) return 32767;
    if (x <= -1.0f) return -32768;
    // Preserve AVFoundation's 32768 scale and half-away-from-zero rounding.
    long v = lroundf(x * 32768.0f);
    return v > 32767 ? 32767 : (int16_t)v;
}
int demucs_write_wav(const char *path, const void *samples, int dtype,
    int64_t frames, int channels, ptrdiff_t sf, ptrdiff_t sc, int sr,
    int float_out, int (*cancelled)(void *), void *context) {
    if (!path || !samples || frames <= 0 || channels <= 0 || channels > 16383 || sr <= 0 ||
        (dtype != 0 && dtype != 1)) return EINVAL;
    uint64_t frame_bytes = (uint64_t)channels * (float_out ? 4u : 2u);
    if ((uint64_t)frames > UINT32_MAX / frame_bytes) return EFBIG;
    uint64_t bytes = (uint64_t)frames * frame_bytes;
    uint32_t overhead = float_out ? 48u : 36u;
    if (bytes > UINT32_MAX - overhead || (uint64_t)sr * channels * (float_out ? 4u : 2u) > UINT32_MAX)
        return EFBIG;
    if (cancelled && cancelled(context)) return ECANCELED;
    FILE *f = fopen(path, "wb");
    if (!f) return errno;
    uint8_t header[56] = {0};
    __builtin_memcpy(header, "RIFF", 4); u32(header + 4, overhead + (uint32_t)bytes);
    __builtin_memcpy(header + 8, "WAVEfmt ", 8); u32(header + 16, 16);
    u16(header + 20, float_out ? 3 : 1); u16(header + 22, channels);
    u32(header + 24, sr); u32(header + 28, (uint32_t)((uint64_t)sr * frame_bytes));
    u16(header + 32, channels * (float_out ? 4 : 2)); u16(header + 34, float_out ? 32 : 16);
    size_t header_size = 44;
    if (float_out) {
        __builtin_memcpy(header + 36, "fact", 4); u32(header + 40, 4); u32(header + 44, (uint32_t)frames);
        __builtin_memcpy(header + 48, "data", 4); u32(header + 52, (uint32_t)bytes);
        header_size = 56;
    } else { __builtin_memcpy(header + 36, "data", 4); u32(header + 40, (uint32_t)bytes); }
    int error = put(f, header, header_size);
    size_t chunk = 16384u / (size_t)channels;
    float *staging = malloc(chunk * channels * sizeof(float));
    int16_t *packed = float_out ? NULL : malloc(chunk * channels * sizeof(int16_t));
    if (!staging || (!float_out && !packed)) error = ENOMEM;
    for (int64_t offset = 0; !error && offset < frames; offset += chunk) {
        if (cancelled && cancelled(context)) { error = ECANCELED; break; }
        size_t count = (size_t)(frames - offset) < chunk ? (size_t)(frames - offset) : chunk;
        if (float_out && dtype == 0 && sc == 1 && sf == channels) {
            error = put(f, (const float *)samples + offset * sf, count * channels * 4);
            continue;
        }
#if defined(__ARM_NEON)
        if (!float_out && dtype == 0 && channels == 2 && sf == 1) {
            const float *left = (const float *)samples + offset;
            const float *right = left + sc;
            size_t i = 0;
            for (; i + 4 <= count; i += 4) {
                float32x4_t l = vld1q_f32(left + i), r = vld1q_f32(right + i);
                l = vbslq_f32(vceqq_f32(l, l), l, vdupq_n_f32(0));
                r = vbslq_f32(vceqq_f32(r, r), r, vdupq_n_f32(0));
                l = vminq_f32(vmaxq_f32(l, vdupq_n_f32(-1)), vdupq_n_f32(1));
                r = vminq_f32(vmaxq_f32(r, vdupq_n_f32(-1)), vdupq_n_f32(1));
                int16x4x2_t lr;
                lr.val[0] = vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(l, 32768)));
                lr.val[1] = vqmovn_s32(vcvtaq_s32_f32(vmulq_n_f32(r, 32768)));
                vst2_s16(packed + i * 2, lr);
            }
            for (; i < count; ++i) { packed[2*i] = pcm16(left[i]); packed[2*i+1] = pcm16(right[i]); }
            error = put(f, packed, count * 4);
            continue;
        }
#endif
        for (size_t i = 0; i < count; ++i) {
            for (int c = 0; c < channels; ++c) {
                float x = sample(samples, dtype, (offset + (ptrdiff_t)i) * sf + c * sc);
                if (float_out) staging[i * channels + c] = x;
                else packed[i * channels + c] = pcm16(x);
            }
        }
        error = put(f, float_out ? (void *)staging : (void *)packed,
            count * channels * (float_out ? 4u : 2u));
    }
    free(staging); free(packed);
    if (fclose(f) != 0 && !error) error = errno ? errno : EIO;
    return error;
}
