// Minimal sherpa-onnx C ABI declarations for HL Meet.
// Source ABI: k2-fsa/sherpa-onnx tag v1.13.2
// Upstream: sherpa-onnx/c-api/c-api.h
// License: Apache-2.0
//
// Only public v1.13.2 symbols used by meeting-speaker-id.cpp are declared.

#ifndef HLMEET_SHERPA_ONNX_C_API_V1_13_2_H_
#define HLMEET_SHERPA_ONNX_C_API_V1_13_2_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SherpaOnnxOnlineStream
    SherpaOnnxOnlineStream;

typedef struct SherpaOnnxSpeakerEmbeddingExtractorConfig {
    const char *model;
    int32_t num_threads;
    int32_t debug;
    const char *provider;
} SherpaOnnxSpeakerEmbeddingExtractorConfig;

typedef struct SherpaOnnxSpeakerEmbeddingExtractor
    SherpaOnnxSpeakerEmbeddingExtractor;

typedef struct SherpaOnnxWave {
    const float *samples;
    int32_t sample_rate;
    int32_t num_samples;
} SherpaOnnxWave;

const SherpaOnnxWave *
SherpaOnnxReadWave(const char *filename);

void SherpaOnnxFreeWave(
    const SherpaOnnxWave *wave
);

const SherpaOnnxSpeakerEmbeddingExtractor *
SherpaOnnxCreateSpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractorConfig *config
);

void SherpaOnnxDestroySpeakerEmbeddingExtractor(
    const SherpaOnnxSpeakerEmbeddingExtractor *p
);

int32_t SherpaOnnxSpeakerEmbeddingExtractorDim(
    const SherpaOnnxSpeakerEmbeddingExtractor *p
);

const SherpaOnnxOnlineStream *
SherpaOnnxSpeakerEmbeddingExtractorCreateStream(
    const SherpaOnnxSpeakerEmbeddingExtractor *p
);

void SherpaOnnxDestroyOnlineStream(
    const SherpaOnnxOnlineStream *stream
);

void SherpaOnnxOnlineStreamAcceptWaveform(
    const SherpaOnnxOnlineStream *stream,
    int32_t sample_rate,
    const float *samples,
    int32_t n
);

void SherpaOnnxOnlineStreamInputFinished(
    const SherpaOnnxOnlineStream *stream
);

int32_t SherpaOnnxSpeakerEmbeddingExtractorIsReady(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s
);

const float *
SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *p,
    const SherpaOnnxOnlineStream *s
);

void SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(
    const float *v
);

#ifdef __cplusplus
}
#endif

#endif
