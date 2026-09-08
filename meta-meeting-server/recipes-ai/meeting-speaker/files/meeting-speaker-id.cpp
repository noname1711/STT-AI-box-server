#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "sherpa-onnx-c-api-v1.13.2.h"

struct Speaker {
    std::string id;
    std::vector<float> centroid;
    std::vector<std::vector<float>> gallery;
    int count = 1;
    size_t gallery_cursor = 0;
};

struct Room {
    std::vector<Speaker> speakers;
    int next_id = 0;
};

static std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> out;
    size_t start = 0;

    while (true) {
        size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }

    return out;
}

static void normalize(std::vector<float> &v) {
    double sum = 0.0;

    for (float x : v)
        sum += static_cast<double>(x) * x;

    const double norm = std::sqrt(sum);

    if (norm <= 1e-12)
        return;

    for (float &x : v)
        x = static_cast<float>(x / norm);
}

static float cosine(
    const std::vector<float> &a,
    const std::vector<float> &b
) {
    if (a.size() != b.size() || a.empty())
        return -1.0f;

    double sum = 0.0;

    for (size_t i = 0; i < a.size(); ++i)
        sum += static_cast<double>(a[i]) * b[i];

    return static_cast<float>(sum);
}

static float speaker_similarity(
    const std::vector<float> &embedding,
    const Speaker &speaker
) {
    float best = cosine(
        embedding,
        speaker.centroid
    );

    for (const auto &sample : speaker.gallery) {
        best = std::max(
            best,
            cosine(embedding, sample)
        );
    }

    return best;
}

static void add_gallery_sample(
    Speaker *speaker,
    const std::vector<float> &embedding
) {
    constexpr size_t kMaxGallery = 8;

    if (speaker->gallery.size() < kMaxGallery) {
        speaker->gallery.push_back(embedding);
        return;
    }

    speaker->gallery[
        speaker->gallery_cursor % kMaxGallery
    ] = embedding;

    speaker->gallery_cursor =
        (speaker->gallery_cursor + 1) % kMaxGallery;
}

static bool compute_embedding(
    const SherpaOnnxSpeakerEmbeddingExtractor *extractor,
    int32_t dim,
    const std::string &wav_path,
    std::vector<float> *embedding,
    std::string *error
) {
    const SherpaOnnxWave *wave =
        SherpaOnnxReadWave(wav_path.c_str());

    if (!wave) {
        *error = "cannot-read-wav";
        return false;
    }

    if (wave->sample_rate != 16000) {
        SherpaOnnxFreeWave(wave);
        *error = "sample-rate-not-16000";
        return false;
    }

    const SherpaOnnxOnlineStream *stream =
        SherpaOnnxSpeakerEmbeddingExtractorCreateStream(
            extractor
        );

    if (!stream) {
        SherpaOnnxFreeWave(wave);
        *error = "cannot-create-stream";
        return false;
    }

    SherpaOnnxOnlineStreamAcceptWaveform(
        stream,
        wave->sample_rate,
        wave->samples,
        wave->num_samples
    );

    SherpaOnnxOnlineStreamInputFinished(stream);

    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(
            extractor,
            stream
        )) {
        SherpaOnnxDestroyOnlineStream(stream);
        SherpaOnnxFreeWave(wave);
        *error = "audio-too-short";
        return false;
    }

    const float *raw =
        SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
            extractor,
            stream
        );

    if (!raw) {
        SherpaOnnxDestroyOnlineStream(stream);
        SherpaOnnxFreeWave(wave);
        *error = "embedding-failed";
        return false;
    }

    embedding->assign(raw, raw + dim);
    normalize(*embedding);

    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(raw);
    SherpaOnnxDestroyOnlineStream(stream);
    SherpaOnnxFreeWave(wave);

    return true;
}

static std::string new_speaker_id(int index) {
    std::ostringstream os;
    os << "SPEAKER_"
       << std::setw(2)
       << std::setfill('0')
       << index;
    return os.str();
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr
            << "usage: meeting-speaker-id "
            << "EMBEDDING_MODEL.onnx [THRESHOLD]\n";
        return 2;
    }

    const std::string model = argv[1];

    float threshold = 0.60f;
    if (argc == 3)
        threshold = std::strtof(argv[2], nullptr);

    if (!(threshold > 0.0f && threshold < 1.0f)) {
        std::cerr << "invalid threshold\n";
        return 2;
    }

    SherpaOnnxSpeakerEmbeddingExtractorConfig config;
    std::memset(&config, 0, sizeof(config));

    config.model = model.c_str();
    config.num_threads = 1;
    config.debug = 0;
    config.provider = "cpu";

    const SherpaOnnxSpeakerEmbeddingExtractor *extractor =
        SherpaOnnxCreateSpeakerEmbeddingExtractor(&config);

    if (!extractor) {
        std::cerr << "cannot create embedding extractor\n";
        return 3;
    }

    const int32_t dim =
        SherpaOnnxSpeakerEmbeddingExtractorDim(extractor);

    if (dim <= 0) {
        std::cerr << "invalid embedding dimension\n";
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor);
        return 3;
    }

    std::unordered_map<std::string, Room> rooms;

    std::cout
        << "READY\tcpu\tanonymous-speaker-embedding"
        << std::endl;

    std::string line;

    while (std::getline(std::cin, line)) {
        if (line == "QUIT")
            break;

        if (line == "PING") {
            std::cout << "OK\tPING" << std::endl;
            continue;
        }

        auto fields = split_tab(line);

        if (
            fields.size() == 2
            && fields[0] == "RESET"
        ) {
            rooms.erase(fields[1]);
            std::cout
                << "OK\tRESET\t"
                << fields[1]
                << std::endl;
            continue;
        }

        if (
            fields.size() != 3
            || fields[0] != "WAV"
        ) {
            std::cout
                << "ERR\tbad-request"
                << std::endl;
            continue;
        }

        const std::string &wav_path = fields[1];
        const std::string &room_name = fields[2];

        std::vector<float> embedding;
        std::string error;

        if (!compute_embedding(
                extractor,
                dim,
                wav_path,
                &embedding,
                &error
            )) {
            std::cout
                << "ERR\t"
                << error
                << std::endl;
            continue;
        }

        Room &room = rooms[room_name];

        int best_index = -1;
        float best_score = -1.0f;
        float second_score = -1.0f;

        for (size_t i = 0; i < room.speakers.size(); ++i) {
            const float score = speaker_similarity(
                embedding,
                room.speakers[i]
            );

            if (score > best_score) {
                second_score = best_score;
                best_score = score;
                best_index = static_cast<int>(i);
            } else if (score > second_score) {
                second_score = score;
            }
        }

        const float soft_threshold =
            std::max(0.35f, threshold - 0.12f);

        const float clear_soft_floor =
            std::max(0.32f, threshold - 0.20f);

        constexpr float kSoftMargin = 0.035f;
        constexpr float kClearSoftMargin = 0.10f;

        const float margin =
            best_index < 0
                ? 0.0f
                : (
                    second_score < -0.5f
                        ? 1.0f
                        : best_score - second_score
                );

        // Preserve the actual nearest-speaker evidence before a NEW
        // decision rewrites the compatibility score to 1.0.
        const float raw_best_score = best_score;
        const float raw_second_score = second_score;
        const float raw_margin = margin;

        bool matched = false;
        std::string match_mode = "new";

        if (best_index >= 0) {
            if (best_score >= threshold) {
                matched = true;
                match_mode = "match-strong";
            } else if (
                best_score >= soft_threshold
                && (
                    room.speakers.size() == 1
                    || margin >= kSoftMargin
                )
            ) {
                matched = true;
                match_mode = "match-soft";
            } else if (
                room.speakers.size() > 1
                && best_score >= clear_soft_floor
                && margin >= kClearSoftMargin
            ) {
                matched = true;
                match_mode = "match-clear-soft";
            }
        }

        const bool is_new = !matched;

        if (is_new) {
            Speaker s;
            s.id = new_speaker_id(room.next_id++);
            s.centroid = embedding;
            s.gallery.push_back(embedding);
            s.count = 1;

            room.speakers.push_back(std::move(s));

            best_index =
                static_cast<int>(room.speakers.size()) - 1;

            best_score = 1.0f;
            second_score = -1.0f;
        } else {
            Speaker &s = room.speakers[best_index];

            const bool trusted_for_update = (
                best_score >= threshold
                || second_score < -0.5f
                || margin >= 0.075f
            );

            if (trusted_for_update) {
                const float alpha =
                    s.count < 4 ? 0.18f : 0.08f;

                for (size_t i = 0; i < s.centroid.size(); ++i) {
                    s.centroid[i] =
                        (1.0f - alpha) * s.centroid[i]
                        + alpha * embedding[i];
                }

                normalize(s.centroid);
                add_gallery_sample(&s, embedding);
            }

            s.count = std::min(
                s.count + 1,
                1000000
            );
        }

        const Speaker &speaker =
            room.speakers[best_index];

        std::cout
            << "OK\t"
            << speaker.id
            << "\t"
            << std::fixed
            << std::setprecision(4)
            << best_score
            << "\t"
            << match_mode
            << "\t"
            << second_score
            << "\t"
            << (
                second_score < -0.5f
                    ? 1.0f
                    : best_score - second_score
            )
            << "	"
            << raw_best_score
            << "	"
            << raw_second_score
            << "	"
            << raw_margin
            << std::endl;
    }

    SherpaOnnxDestroySpeakerEmbeddingExtractor(
        extractor
    );

    return 0;
}
