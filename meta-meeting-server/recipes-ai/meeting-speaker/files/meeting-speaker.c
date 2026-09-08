#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static int32_t progress_callback(
    int32_t processed,
    int32_t total,
    void *arg) {

    (void)arg;

    if (total > 0) {
        float p = 100.0f * processed / total;
        fprintf(stderr, "DIARIZATION_PROGRESS=%.1f%%\n", p);
    }

    return 0;
}

static void usage(const char *prog) {
    fprintf(
        stderr,
        "Usage:\n"
        "  %s SEGMENTATION.onnx EMBEDDING.onnx AUDIO.wav "
        "[NUM_SPEAKERS] [THRESHOLD]\n\n"
        "Examples:\n"
        "  %s segmentation.onnx embedding.onnx meeting.wav 2\n"
        "  %s segmentation.onnx embedding.onnx meeting.wav 0 0.5\n",
        prog, prog, prog);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return 2;
    }

    const char *segmentation_model = argv[1];
    const char *embedding_model = argv[2];
    const char *wav_filename = argv[3];

    int32_t num_speakers = 0;
    float threshold = 0.5f;

    if (argc >= 5) {
        num_speakers = (int32_t)strtol(argv[4], NULL, 10);
        if (num_speakers < 0) {
            fprintf(stderr, "ERROR: NUM_SPEAKERS must be >= 0\n");
            return 2;
        }
    }

    if (argc >= 6) {
        threshold = strtof(argv[5], NULL);

        if (threshold <= 0.0f || threshold >= 1.0f) {
            fprintf(
                stderr,
                "ERROR: THRESHOLD must be between 0 and 1\n");
            return 2;
        }
    }

    const SherpaOnnxWave *wave =
        SherpaOnnxReadWave(wav_filename);

    if (!wave) {
        fprintf(
            stderr,
            "ERROR: failed to read WAV: %s\n",
            wav_filename);
        return 3;
    }

    SherpaOnnxOfflineSpeakerDiarizationConfig config;
    memset(&config, 0, sizeof(config));

    config.segmentation.pyannote.model =
        segmentation_model;

    config.embedding.model =
        embedding_model;

    if (num_speakers > 0) {
        config.clustering.num_clusters =
            num_speakers;
    } else {
        config.clustering.threshold =
            threshold;
    }

    const SherpaOnnxOfflineSpeakerDiarization *sd =
        SherpaOnnxCreateOfflineSpeakerDiarization(
            &config);

    if (!sd) {
        fprintf(
            stderr,
            "ERROR: unable to initialize speaker diarization\n");

        SherpaOnnxFreeWave(wave);
        return 4;
    }

    int32_t required_rate =
        SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(sd);

    fprintf(
        stderr,
        "MODEL_SAMPLE_RATE=%d\n"
        "AUDIO_SAMPLE_RATE=%d\n"
        "AUDIO_SAMPLES=%d\n",
        required_rate,
        wave->sample_rate,
        wave->num_samples);

    if (wave->sample_rate != required_rate) {
        fprintf(
            stderr,
            "ERROR: sample-rate mismatch\n");

        SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
        SherpaOnnxFreeWave(wave);
        return 5;
    }

    const SherpaOnnxOfflineSpeakerDiarizationResult *result =
        SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
            sd,
            wave->samples,
            wave->num_samples,
            progress_callback,
            NULL);

    if (!result) {
        fprintf(
            stderr,
            "ERROR: diarization inference failed\n");

        SherpaOnnxDestroyOfflineSpeakerDiarization(sd);
        SherpaOnnxFreeWave(wave);
        return 6;
    }

    int32_t num_segments =
        SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
            result);

    int32_t detected_speakers =
        SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
            result);

    const SherpaOnnxOfflineSpeakerDiarizationSegment *segments =
        SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
            result);

    printf(
        "{\"type\":\"summary\","
        "\"speakers\":%d,"
        "\"segments\":%d}\n",
        detected_speakers,
        num_segments);

    for (int32_t i = 0; i < num_segments; ++i) {
        printf(
            "{\"type\":\"segment\","
            "\"start\":%.3f,"
            "\"end\":%.3f,"
            "\"speaker\":\"SPEAKER_%02d\"}\n",
            segments[i].start,
            segments[i].end,
            segments[i].speaker);
    }

    fflush(stdout);

    if (segments) {
        SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
            segments);
    }

    SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
        result);

    SherpaOnnxDestroyOfflineSpeakerDiarization(
        sd);

    SherpaOnnxFreeWave(
        wave);

    return 0;
}
