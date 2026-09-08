#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sherpa-onnx-c-api-v1.13.2.h"
#include "sherpa-onnx-diarization-abi-v1.13.2.h"

namespace {

constexpr int32_t kSampleRate = 16000;
constexpr int32_t kMinCleanPieceSamples = kSampleRate * 35 / 100;
constexpr int32_t kMinEmbeddingSamples = kSampleRate;
constexpr int32_t kMaxEmbeddingSamples = kSampleRate * 8;

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

struct Range {
    int32_t start = 0;
    int32_t end = 0;
};

struct LocalSpeaker {
    int32_t label = -1;
    std::vector<Range> ranges;
    int32_t clean_samples = 0;
    std::vector<float> embedding;
};

struct Decision {
    bool valid = false;
    std::string id = "UNKNOWN";
    std::string mode = "unknown";
    float score = -1.0f;
    float second = -1.0f;
    float margin = 0.0f;
    float raw_best = -1.0f;
    float raw_second = -1.0f;
    float raw_margin = 0.0f;
};

std::vector<std::string> split_tab(const std::string &line) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            break;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

void normalize(std::vector<float> &v) {
    double sum = 0.0;
    for (float x : v)
        sum += static_cast<double>(x) * x;
    const double norm = std::sqrt(sum);
    if (norm <= 1e-12)
        return;
    for (float &x : v)
        x = static_cast<float>(x / norm);
}

float cosine(
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

float speaker_similarity(
    const std::vector<float> &embedding,
    const Speaker &speaker
) {
    float best = cosine(embedding, speaker.centroid);
    for (const auto &sample : speaker.gallery)
        best = std::max(best, cosine(embedding, sample));
    return best;
}

void add_gallery_sample(
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

std::string new_speaker_id(int index) {
    std::ostringstream os;
    os << "SPEAKER_"
       << std::setw(2)
       << std::setfill('0')
       << index;
    return os.str();
}

int32_t noop_progress(int32_t, int32_t, void *) {
    return 0;
}

bool compute_embedding_samples(
    const SherpaOnnxSpeakerEmbeddingExtractor *extractor,
    int32_t dim,
    const float *samples,
    int32_t n,
    std::vector<float> *embedding
) {
    if (!extractor || !samples || n <= 0 || !embedding)
        return false;

    const SherpaOnnxOnlineStream *stream =
        SherpaOnnxSpeakerEmbeddingExtractorCreateStream(extractor);
    if (!stream)
        return false;

    SherpaOnnxOnlineStreamAcceptWaveform(
        stream,
        kSampleRate,
        samples,
        n
    );
    SherpaOnnxOnlineStreamInputFinished(stream);

    if (!SherpaOnnxSpeakerEmbeddingExtractorIsReady(
            extractor,
            stream
        )) {
        SherpaOnnxDestroyOnlineStream(stream);
        return false;
    }

    const float *raw =
        SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding(
            extractor,
            stream
        );
    if (!raw) {
        SherpaOnnxDestroyOnlineStream(stream);
        return false;
    }

    embedding->assign(raw, raw + dim);
    normalize(*embedding);

    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding(raw);
    SherpaOnnxDestroyOnlineStream(stream);
    return true;
}

std::vector<Range> merge_ranges(std::vector<Range> ranges) {
    ranges.erase(
        std::remove_if(
            ranges.begin(),
            ranges.end(),
            [](const Range &r) { return r.end <= r.start; }
        ),
        ranges.end()
    );

    std::sort(
        ranges.begin(),
        ranges.end(),
        [](const Range &a, const Range &b) {
            if (a.start != b.start)
                return a.start < b.start;
            return a.end < b.end;
        }
    );

    std::vector<Range> out;
    for (const Range &r : ranges) {
        if (out.empty() || r.start > out.back().end) {
            out.push_back(r);
        } else {
            out.back().end =
                std::max(out.back().end, r.end);
        }
    }
    return out;
}

std::vector<Range> subtract_ranges(
    Range base,
    std::vector<Range> blocked
) {
    blocked = merge_ranges(std::move(blocked));
    std::vector<Range> out;
    int32_t cursor = base.start;

    for (const Range &b : blocked) {
        if (b.end <= cursor || b.start >= base.end)
            continue;

        const int32_t bs = std::max(base.start, b.start);
        const int32_t be = std::min(base.end, b.end);

        if (bs > cursor)
            out.push_back({cursor, bs});

        cursor = std::max(cursor, be);
        if (cursor >= base.end)
            break;
    }

    if (cursor < base.end)
        out.push_back({cursor, base.end});

    out.erase(
        std::remove_if(
            out.begin(),
            out.end(),
            [](const Range &r) {
                return r.end - r.start < kMinCleanPieceSamples;
            }
        ),
        out.end()
    );
    return out;
}

std::vector<float> collect_samples(
    const SherpaOnnxWave *wave,
    const std::vector<Range> &ranges
) {
    std::vector<float> out;
    if (!wave)
        return out;

    out.reserve(
        std::min<int32_t>(
            kMaxEmbeddingSamples,
            wave->num_samples
        )
    );

    for (const Range &r : ranges) {
        if (static_cast<int32_t>(out.size()) >= kMaxEmbeddingSamples)
            break;

        const int32_t start =
            std::max<int32_t>(0, std::min(r.start, wave->num_samples));
        const int32_t end =
            std::max<int32_t>(start, std::min(r.end, wave->num_samples));
        int32_t count = end - start;

        count = std::min<int32_t>(
            count,
            kMaxEmbeddingSamples - static_cast<int32_t>(out.size())
        );

        if (count <= 0)
            continue;

        out.insert(
            out.end(),
            wave->samples + start,
            wave->samples + start + count
        );
    }

    return out;
}

Decision assign_global(
    Room *room,
    const std::vector<float> &embedding,
    float threshold,
    bool allow_single_soft
) {
    Decision d;
    if (!room || embedding.empty())
        return d;

    int best_index = -1;
    float best_score = -1.0f;
    float second_score = -1.0f;

    for (size_t i = 0; i < room->speakers.size(); ++i) {
        const float score =
            speaker_similarity(embedding, room->speakers[i]);

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

    d.raw_best = best_score;
    d.raw_second = second_score;
    d.raw_margin = margin;

    bool matched = false;
    std::string match_mode = "new";

    if (best_index >= 0) {
        if (best_score >= threshold) {
            matched = true;
            match_mode = "match-strong";
        } else if (
            best_score >= soft_threshold
            && (
                (
                    room->speakers.size() == 1
                    && allow_single_soft
                )
                || (
                    room->speakers.size() > 1
                    && margin >= kSoftMargin
                )
            )
        ) {
            matched = true;
            match_mode = "match-soft";
        } else if (
            room->speakers.size() > 1
            && best_score >= clear_soft_floor
            && margin >= kClearSoftMargin
        ) {
            matched = true;
            match_mode = "match-clear-soft";
        }
    }

    if (!matched) {
        Speaker s;
        s.id = new_speaker_id(room->next_id++);
        s.centroid = embedding;
        s.gallery.push_back(embedding);
        s.count = 1;

        room->speakers.push_back(std::move(s));
        best_index =
            static_cast<int>(room->speakers.size()) - 1;

        d.score = 1.0f;
        d.second = -1.0f;
        d.margin = 1.0f;
        d.mode = "new";
    } else {
        Speaker &s = room->speakers[best_index];

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

        s.count = std::min(s.count + 1, 1000000);

        d.score = best_score;
        d.second = second_score;
        d.margin = margin;
        d.mode = match_mode;
    }

    d.valid = true;
    d.id = room->speakers[best_index].id;
    return d;
}

std::string make_json(
    int32_t detected_speakers,
    const SherpaOnnxOfflineSpeakerDiarizationSegment *segments,
    int32_t num_segments,
    const std::map<int32_t, Decision> &decisions,
    const std::map<int32_t, int32_t> &clean_samples,
    const std::string &dominant
) {
    std::ostringstream os;
    os << "{\"engine\":\"seg-reid-v1\""
       << ",\"diarized_speakers\":" << detected_speakers
       << ",\"dominant\":\"" << dominant << "\""
       << ",\"regions\":[";

    for (int32_t i = 0; i < num_segments; ++i) {
        if (i)
            os << ",";

        const int32_t local = segments[i].speaker;
        const auto it = decisions.find(local);
        const std::string global =
            (
                it != decisions.end()
                && it->second.valid
            )
                ? it->second.id
                : "UNKNOWN";

        os << "{\"start\":"
           << std::fixed << std::setprecision(3)
           << segments[i].start
           << ",\"end\":"
           << segments[i].end
           << ",\"local\":" << local
           << ",\"speaker_id\":\""
           << global
           << "\"}";
    }

    os << "],\"locals\":[";
    bool first = true;

    for (const auto &item : clean_samples) {
        if (!first)
            os << ",";
        first = false;

        const int32_t local = item.first;
        const auto it = decisions.find(local);
        const bool valid =
            it != decisions.end() && it->second.valid;

        os << "{\"local\":" << local
           << ",\"clean_ms\":"
           << (item.second * 1000 / kSampleRate)
           << ",\"speaker_id\":\""
           << (valid ? it->second.id : "UNKNOWN")
           << "\"";

        if (valid) {
            os << ",\"score\":"
               << std::fixed << std::setprecision(4)
               << it->second.score
               << ",\"match\":\""
               << it->second.mode
               << "\""
               << ",\"raw_best\":"
               << it->second.raw_best
               << ",\"raw_second\":"
               << it->second.raw_second
               << ",\"raw_margin\":"
               << it->second.raw_margin;
        }

        os << "}";
    }

    os << "]}";
    return os.str();
}

void print_ok(
    const Decision &d,
    const std::string &mode_prefix,
    const std::string &json
) {
    std::cout
        << "OK\t"
        << d.id
        << "\t"
        << std::fixed
        << std::setprecision(4)
        << d.score
        << "\t"
        << mode_prefix
        << d.mode
        << "\t"
        << d.second
        << "\t"
        << d.margin
        << "\t"
        << d.raw_best
        << "\t"
        << d.raw_second
        << "\t"
        << d.raw_margin
        << "\t"
        << json
        << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr
            << "usage: meeting-speaker-seg-reid "
            << "SEGMENTATION.onnx EMBEDDING.onnx "
            << "[REID_THRESHOLD] [DIARIZATION_THRESHOLD]\n";
        return 2;
    }

    const std::string segmentation_model = argv[1];
    const std::string embedding_model = argv[2];

    float reid_threshold = 0.60f;
    float diar_threshold = 0.50f;

    if (argc >= 4)
        reid_threshold = std::strtof(argv[3], nullptr);
    if (argc >= 5)
        diar_threshold = std::strtof(argv[4], nullptr);

    if (!(reid_threshold > 0.0f && reid_threshold < 1.0f)) {
        std::cerr << "invalid reid threshold\n";
        return 2;
    }

    if (!(diar_threshold > 0.0f && diar_threshold < 1.0f)) {
        std::cerr << "invalid diarization threshold\n";
        return 2;
    }

    SherpaOnnxSpeakerEmbeddingExtractorConfig embed_config;
    std::memset(&embed_config, 0, sizeof(embed_config));
    embed_config.model = embedding_model.c_str();
    embed_config.num_threads = 1;
    embed_config.debug = 0;
    embed_config.provider = "cpu";

    const SherpaOnnxSpeakerEmbeddingExtractor *extractor =
        SherpaOnnxCreateSpeakerEmbeddingExtractor(&embed_config);

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

    SherpaOnnxOfflineSpeakerDiarizationConfig diar_config;
    std::memset(&diar_config, 0, sizeof(diar_config));

    diar_config.segmentation.pyannote.model =
        segmentation_model.c_str();
    diar_config.segmentation.num_threads = 1;
    diar_config.segmentation.debug = 0;
    diar_config.segmentation.provider = "cpu";

    diar_config.embedding.model =
        embedding_model.c_str();
    diar_config.embedding.num_threads = 1;
    diar_config.embedding.debug = 0;
    diar_config.embedding.provider = "cpu";

    diar_config.clustering.num_clusters = 0;
    diar_config.clustering.threshold = diar_threshold;
    diar_config.min_duration_on = 0.0f;
    diar_config.min_duration_off = 0.0f;

    const SherpaOnnxOfflineSpeakerDiarization *diarizer =
        SherpaOnnxCreateOfflineSpeakerDiarization(&diar_config);

    if (!diarizer) {
        std::cerr << "cannot create offline diarizer\n";
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor);
        return 4;
    }

    if (
        SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(
            diarizer
        ) != kSampleRate
    ) {
        std::cerr << "unexpected diarizer sample rate\n";
        SherpaOnnxDestroyOfflineSpeakerDiarization(diarizer);
        SherpaOnnxDestroySpeakerEmbeddingExtractor(extractor);
        return 4;
    }

    std::unordered_map<std::string, Room> rooms;

    std::cout
        << "READY\tcpu\tsegmentation-reid-v1"
        << std::endl;

    std::string line;

    while (std::getline(std::cin, line)) {
        if (line == "QUIT")
            break;

        if (line == "PING") {
            std::cout << "OK\tPING" << std::endl;
            continue;
        }

        const auto fields = split_tab(line);

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
            std::cout << "ERR\tbad-request" << std::endl;
            continue;
        }

        const std::string &wav_path = fields[1];
        const std::string &room_name = fields[2];

        const SherpaOnnxWave *wave =
            SherpaOnnxReadWave(wav_path.c_str());

        if (!wave) {
            std::cout << "ERR\tcannot-read-wav" << std::endl;
            continue;
        }

        if (
            wave->sample_rate != kSampleRate
            || wave->num_samples <= 0
        ) {
            SherpaOnnxFreeWave(wave);
            std::cout << "ERR\tinvalid-wave" << std::endl;
            continue;
        }

        const SherpaOnnxOfflineSpeakerDiarizationResult *result =
            SherpaOnnxOfflineSpeakerDiarizationProcessWithCallback(
                diarizer,
                wave->samples,
                wave->num_samples,
                noop_progress,
                nullptr
            );

        if (!result) {
            SherpaOnnxFreeWave(wave);
            std::cout << "ERR\tdiarization-failed" << std::endl;
            continue;
        }

        const int32_t detected_speakers =
            SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(
                result
            );

        const int32_t num_segments =
            SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(
                result
            );

        const SherpaOnnxOfflineSpeakerDiarizationSegment *segments =
            num_segments > 0
                ? SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(
                    result
                )
                : nullptr;

        std::map<int32_t, LocalSpeaker> locals;

        for (int32_t i = 0; i < num_segments; ++i) {
            const int32_t start = std::max<int32_t>(
                0,
                std::min<int32_t>(
                    wave->num_samples,
                    static_cast<int32_t>(
                        std::floor(
                            segments[i].start * kSampleRate
                        )
                    )
                )
            );

            const int32_t end = std::max<int32_t>(
                start,
                std::min<int32_t>(
                    wave->num_samples,
                    static_cast<int32_t>(
                        std::ceil(
                            segments[i].end * kSampleRate
                        )
                    )
                )
            );

            if (end <= start)
                continue;

            std::vector<Range> blocked;

            for (int32_t j = 0; j < num_segments; ++j) {
                if (
                    i == j
                    || segments[j].speaker == segments[i].speaker
                ) {
                    continue;
                }

                const int32_t other_start = std::max<int32_t>(
                    0,
                    std::min<int32_t>(
                        wave->num_samples,
                        static_cast<int32_t>(
                            std::floor(
                                segments[j].start * kSampleRate
                            )
                        )
                    )
                );

                const int32_t other_end = std::max<int32_t>(
                    other_start,
                    std::min<int32_t>(
                        wave->num_samples,
                        static_cast<int32_t>(
                            std::ceil(
                                segments[j].end * kSampleRate
                            )
                        )
                    )
                );

                const int32_t os =
                    std::max(start, other_start);
                const int32_t oe =
                    std::min(end, other_end);

                if (oe > os)
                    blocked.push_back({os, oe});
            }

            auto clean = subtract_ranges(
                {start, end},
                std::move(blocked)
            );

            LocalSpeaker &local =
                locals[segments[i].speaker];

            local.label = segments[i].speaker;

            local.ranges.insert(
                local.ranges.end(),
                clean.begin(),
                clean.end()
            );
        }

        for (auto &item : locals) {
            LocalSpeaker &local = item.second;

            local.ranges =
                merge_ranges(std::move(local.ranges));

            local.clean_samples = 0;

            for (const Range &r : local.ranges)
                local.clean_samples += r.end - r.start;

            const std::vector<float> clean_audio =
                collect_samples(wave, local.ranges);

            if (
                static_cast<int32_t>(clean_audio.size())
                >= kMinEmbeddingSamples
            ) {
                compute_embedding_samples(
                    extractor,
                    dim,
                    clean_audio.data(),
                    static_cast<int32_t>(clean_audio.size()),
                    &local.embedding
                );
            }
        }

        Room &room = rooms[room_name];
        std::map<int32_t, Decision> decisions;
        std::map<int32_t, int32_t> clean_samples;

        int valid_locals = 0;
        for (const auto &item : locals) {
            if (!item.second.embedding.empty())
                ++valid_locals;
        }

        for (const auto &item : locals) {
            const LocalSpeaker &local = item.second;

            clean_samples[item.first] =
                local.clean_samples;

            if (local.embedding.empty())
                continue;

            const bool allow_single_soft =
                valid_locals <= 1;

            decisions[item.first] =
                assign_global(
                    &room,
                    local.embedding,
                    reid_threshold,
                    allow_single_soft
                );
        }

        int32_t dominant_local = -1;
        int32_t dominant_samples = -1;

        for (const auto &item : locals) {
            const auto dit = decisions.find(item.first);

            if (
                dit == decisions.end()
                || !dit->second.valid
            ) {
                continue;
            }

            if (item.second.clean_samples > dominant_samples) {
                dominant_samples =
                    item.second.clean_samples;
                dominant_local =
                    item.first;
            }
        }

        if (dominant_local < 0) {
            if (detected_speakers <= 1) {
                std::vector<float> fallback;

                if (
                    compute_embedding_samples(
                        extractor,
                        dim,
                        wave->samples,
                        wave->num_samples,
                        &fallback
                    )
                ) {
                    Decision d =
                        assign_global(
                            &room,
                            fallback,
                            reid_threshold,
                            true
                        );

                    const std::string json =
                        "{\"engine\":\"seg-reid-v1\","
                        "\"fallback\":\"whole-single-speaker\","
                        "\"regions\":[]}";

                    print_ok(
                        d,
                        "seg-fallback-",
                        json
                    );
                } else {
                    std::cout
                        << "ERR\tinsufficient-speaker-audio"
                        << std::endl;
                }
            } else {
                std::cout
                    << "ERR\tambiguous-overlap"
                    << std::endl;
            }

            if (segments) {
                SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
                    segments
                );
            }

            SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
                result
            );

            SherpaOnnxFreeWave(wave);
            continue;
        }

        const Decision dominant =
            decisions[dominant_local];

        const std::string json =
            make_json(
                detected_speakers,
                segments,
                num_segments,
                decisions,
                clean_samples,
                dominant.id
            );

        print_ok(
            dominant,
            "seg-",
            json
        );

        if (segments) {
            SherpaOnnxOfflineSpeakerDiarizationDestroySegment(
                segments
            );
        }

        SherpaOnnxOfflineSpeakerDiarizationDestroyResult(
            result
        );

        SherpaOnnxFreeWave(wave);
    }

    SherpaOnnxDestroyOfflineSpeakerDiarization(
        diarizer
    );

    SherpaOnnxDestroySpeakerEmbeddingExtractor(
        extractor
    );

    return 0;
}
