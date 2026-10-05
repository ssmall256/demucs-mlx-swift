// Direct RIFF PCM decoding adapted from mlx-audio-io's Apple WAV fast path.
// Copyright (c) 2025 ssmall256. MIT License; see the repository LICENSE.
// This variant validates chunk extents and fills caller-owned planar storage.
#include "CDemucsAudioIO.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

struct demucs_pcm_wav {
    FILE *file;
    demucs_pcm_wav_info info;
    int block_align;
    off_t data_offset;
};
static uint16_t u16(const unsigned char *p) {
    return (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
}
static uint32_t u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
        ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
void demucs_close_pcm_wav(demucs_pcm_wav *reader) {
    if (reader) { fclose(reader->file); free(reader); }
}
int demucs_open_pcm_wav(const char *path, int sample_rate,
        demucs_pcm_wav **result, demucs_pcm_wav_info *info) {
    if (!path || !result || !info || sample_rate <= 0) return -EINVAL;
    *result = NULL;
    FILE *file = fopen(path, "rb");
    if (!file) return -errno;
    int status = 0;
    struct stat st;
    unsigned char header[12];
    demucs_pcm_wav_info parsed = {0};
    int block_align = 0, found_fmt = 0;
    if (fstat(fileno(file), &st) || st.st_size < 12 ||
        fread(header, 1, 12, file) != 12 ||
        memcmp(header, "RIFF", 4) || memcmp(header + 8, "WAVE", 4)) goto done;
    uint64_t end = (uint64_t)u32(header + 4) + 8;
    if (end < 12 || end > (uint64_t)st.st_size) goto done;
    uint64_t position = 12;
    while (position + 8 <= end) {
        unsigned char chunk[8];
        if (fseeko(file, (off_t)position, SEEK_SET) || fread(chunk, 1, 8, file) != 8) goto done;
        uint64_t size = u32(chunk + 4), start = position + 8;
        if (size > end - start || size + (size & 1) > end - start) goto done;
        if (!memcmp(chunk, "fmt ", 4)) {
            unsigned char fmt[40];
            if (found_fmt || size < 16 || fread(fmt, 1, 16, file) != 16) goto done;
            int tag = u16(fmt), channels = u16(fmt + 2), bits = u16(fmt + 14);
            uint32_t rate = u32(fmt + 4);
            if (tag == 0xfffe) {
                static const unsigned char base[12] = {
                    0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71};
                if (size < 40 || fread(fmt + 16, 1, 24, file) != 24 ||
                    u16(fmt + 16) < 22 || u16(fmt + 16) > size - 18 ||
                    u16(fmt + 18) != bits || memcmp(fmt + 28, base, 12)) goto done;
                uint32_t subtype = u32(fmt + 24);
                if (subtype != 1 && subtype != 3) goto done;
                tag = (int)subtype;
            }
            if (!channels || !rate || rate > INT_MAX || (int)rate != sample_rate ||
                !((tag == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) ||
                  (tag == 3 && bits == 32))) goto done;
            block_align = channels * (bits / 8);
            if (block_align > UINT16_MAX || u16(fmt + 12) != block_align ||
                (uint64_t)u32(fmt + 8) != (uint64_t)rate * block_align) goto done;
            parsed.sample_rate = (int)rate;
            parsed.channels = channels;
            parsed.bits_per_sample = bits;
            parsed.is_float = tag == 3;
            found_fmt = 1;
        } else if (!memcmp(chunk, "data", 4)) {
            if (!found_fmt || size % block_align || size / block_align < 2) goto done;
            parsed.frames = (int64_t)(size / block_align);
            demucs_pcm_wav *reader = malloc(sizeof(*reader));
            if (!reader) { status = -ENOMEM; goto done; }
            reader->file = file;
            reader->info = parsed;
            reader->block_align = block_align;
            reader->data_offset = (off_t)start;
            *result = reader;
            *info = parsed;
            return 1;
        }
        position = start + size + (size & 1);
    }
done:
    fclose(file);
    return status;
}
int demucs_read_pcm_wav(demucs_pcm_wav *reader, float *planes,
        int64_t capacity, ptrdiff_t plane_stride,
        int (*cancelled)(void *), void *context) {
    if (!reader || !planes || capacity < reader->info.frames ||
        plane_stride < capacity || plane_stride > PTRDIFF_MAX / 4 / reader->info.channels)
        return EINVAL;
    const demucs_pcm_wav_info info = reader->info;
    // Scratch stays below 256 KiB, independent of track duration/channel count.
    size_t chunk_frames = 262144 / reader->block_align;
    if (chunk_frames > 65536) chunk_frames = 65536;
    const int direct_float = info.is_float && info.channels == 1;
    unsigned char *bytes = direct_float ? NULL : malloc(chunk_frames * reader->block_align);
    if (!direct_float && !bytes) return ENOMEM;
    int error = 0;
    if (fseeko(reader->file, reader->data_offset, SEEK_SET)) { error = errno; goto done; }
    for (int64_t start = 0; start < info.frames;) {
        if (cancelled && cancelled(context)) { error = ECANCELED; goto done; }
        size_t n = (size_t)(info.frames - start);
        if (n > chunk_frames) n = chunk_frames;
        size_t count = n * reader->block_align;
        if (direct_float) {
            if (fread(planes + start, 1, count, reader->file) != count) { error = EIO; goto done; }
            start += (int64_t)n;
            continue;
        }
        if (fread(bytes, 1, count, reader->file) != count) { error = EIO; goto done; }
        for (int c = 0; c < info.channels; ++c) {
            float *output = planes + c * plane_stride + start;
            const unsigned char *input = bytes + c * (info.bits_per_sample / 8);
            size_t i = 0;
#if defined(__aarch64__)
            // Common mono/stereo layouts: deinterleave and convert vectors in
            // one pass. Power-of-two scaling preserves scalar Float32 values.
            if (info.channels <= 2 && !info.is_float && info.bits_per_sample == 16) {
                for (; i + 8 <= n; i += 8) {
                    const int16_t *p = (const int16_t *)bytes + i * info.channels;
                    int16x8_t v = info.channels == 1 ? vld1q_s16(p) : vld2q_s16(p).val[c];
                    vst1q_f32(output + i, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))), 1.0f / 32768.0f));
                    vst1q_f32(output + i + 4, vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v))), 1.0f / 32768.0f));
                }
            } else if (info.channels <= 2 && info.bits_per_sample == 24) {
                const size_t frames_per_vector = info.channels == 1 ? 8 : 4;
                for (; i + frames_per_vector <= n; i += frames_per_vector) {
                    uint8x8x3_t b = vld3_u8(bytes + i * reader->block_align);
                    uint16x8_t low = vorrq_u16(vmovl_u8(b.val[0]),
                        vshlq_n_u16(vmovl_u8(b.val[1]), 8));
                    int16x8_t high = vmovl_s8(vreinterpret_s8_u8(b.val[2]));
                    int32x4_t x = vreinterpretq_s32_u32(vorrq_u32(vmovl_u16(vget_low_u16(low)),
                        vreinterpretq_u32_s32(vshlq_n_s32(vmovl_s16(vget_low_s16(high)), 16))));
                    int32x4_t y = vreinterpretq_s32_u32(vorrq_u32(vmovl_u16(vget_high_u16(low)),
                        vreinterpretq_u32_s32(vshlq_n_s32(vmovl_s16(vget_high_s16(high)), 16))));
                    float32x4_t a = vmulq_n_f32(vcvtq_f32_s32(x), 1.0f / 8388608.0f);
                    float32x4_t bfloat = vmulq_n_f32(vcvtq_f32_s32(y), 1.0f / 8388608.0f);
                    if (info.channels == 1) {
                        vst1q_f32(output + i, a);
                        vst1q_f32(output + i + 4, bfloat);
                    } else {
                        vst1q_f32(output + i, c == 0 ? vuzp1q_f32(a, bfloat) : vuzp2q_f32(a, bfloat));
                    }
                }
            } else if (info.channels <= 2 && info.bits_per_sample == 32) {
                for (; i + 4 <= n; i += 4) {
                    const int32_t *p = (const int32_t *)bytes + i * info.channels;
                    int32x4_t v = info.channels == 1 ? vld1q_s32(p) : vld2q_s32(p).val[c];
                    float32x4_t value = info.is_float ? vreinterpretq_f32_s32(v)
                        : vmulq_n_f32(vcvtq_f32_s32(v), 1.0f / 2147483648.0f);
                    vst1q_f32(output + i, value);
                }
            }
#endif
            input += i * reader->block_align;
            for (; i < n; ++i, input += reader->block_align) {
                if (info.is_float) {
                    uint32_t word = u32(input);
                    memcpy(output + i, &word, 4);
                } else if (info.bits_per_sample == 8) {
                    output[i] = ((int)input[0] - 128) * (1.0f / 128.0f);
                } else if (info.bits_per_sample == 16) {
                    output[i] = (int16_t)u16(input) * (1.0f / 32768.0f);
                } else if (info.bits_per_sample == 24) {
                    int32_t value = (int32_t)((uint32_t)input[0] << 8 |
                        (uint32_t)input[1] << 16 | (uint32_t)input[2] << 24);
                    output[i] = (value >> 8) * (1.0f / 8388608.0f);
                } else {
                    output[i] = (int32_t)u32(input) * (1.0f / 2147483648.0f);
                }
            }
        }
        start += (int64_t)n;
    }
    if (cancelled && cancelled(context)) error = ECANCELED;
done:
    free(bytes);
    return error;
}
