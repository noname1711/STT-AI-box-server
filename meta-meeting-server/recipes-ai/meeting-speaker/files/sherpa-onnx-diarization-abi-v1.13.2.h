// Minimal sherpa-onnx v1.13.2 offline diarization ABI for HL Meet.
// Exact public ABI pinned to k2-fsa/sherpa-onnx tag v1.13.2.
// License: Apache-2.0.

#ifndef HLMEET_SHERPA_ONNX_DIARIZATION_ABI_V1_13_2_H_
#define HLMEET_SHERPA_ONNX_DIARIZATION_ABI_V1_13_2_H_

#include <stdint.h>
#include "sherpa-onnx-c-api-v1.13.2.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig {
    const char *model;
} SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig;

typedef struct SherpaOnnxOfflineSpeakerSegmentationModelConfig {
    SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig pyannote;
    int32_t num_threads;
    int32_t debug;
    const char *provider;
} SherpaOnnxOfflineSpeakerSegmentationModelConfig;

typedef struct SherpaOnnxFastClusteringConfig {
    int32_t num_clusters;
    float threshold;
} SherpaOnnxFastClusteringConfig;

typedef struct SherpaOnnxOfflineSpeakerDiarizationConfig {
    SherpaOnnxOfflineSpeakerSegmentationModelConfig segmentation;
    SherpaOnnxSpeakerEmbeddingExtractorConfig embedding;
    SherpaOnnxFastClusteringConfig clustering;
    float min_duration_on;
    float min_duration_off;
} SherpaOnnxOfflineSpeakerDiarizationConfig;

typedef struct SherpaOnnxOfflineSpeakerDiarization
    SherpaOnnxOfflineSpeakerDiarization;

typedef struct SherpaOnnxOfflineSpeakerDiarizationResult
    SherpaOnnxOfflineSpeakerDiarizationResult;

typedef struct SherpaOnnxOfflineSpeakerDiarizationSegment {
    float start;
    float end;
    int32_t speaker;
} SherpaOnnxOfflineSpeakerDiarizationSegment;

typedef int32_t (*SherpaOnnxOfflineSpeakerDiarizationProgressCallback)(
    int32_t num_processed_chunks,
    int32_t num_total_chunks,
    void *arg
);

const SherpaOnnxOfflineSpeakerDiarization *
SherpaOnnxCreateOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarizationConfig *config
);

void SherpaOnnxDestroyOfflineSpeakerDiarization(
    const SherpaOnnxOfflineSpeakerDiarization *sd
);

int32_t SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
    const SherpaOnnxOfflineSpeakerDiarization *sd
);

const SherpaOnnxOfflineSpeakerDiarizationResult *
SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
    const SherpaOnnxOfflineSpeakerDiarization *sd,
    const float *samples,
    int32_t n,
    SherpaOnnxOfflineSpeakerDiarizationProgressCallback callback,
    void *arg
);

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r
);

int32_t SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r
);

const SherpaOnnxOfflineSpeakerDiarizationSegment *
SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r
);

void SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
    const SherpaOnnxOfflineSpeakerDiarizationSegment *s
);

void SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
    const SherpaOnnxOfflineSpeakerDiarizationResult *r
);

#ifdef __cplusplus
}
#endif

#endif
