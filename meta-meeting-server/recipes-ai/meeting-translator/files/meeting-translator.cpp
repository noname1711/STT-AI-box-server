#include <ctranslate2/devices.h>
#include <ctranslate2/translator.h>
#include <ctranslate2/types.h>
#include <sentencepiece_processor.h>

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct Engine {
    sentencepiece::SentencePieceProcessor sp;
    std::unique_ptr<ctranslate2::Translator> translator;
};

static std::string one_line(std::string s) {
    for (char &c : s) {
        if (c == '\n' || c == '\r' || c == '\t')
            c = ' ';
    }
    return s;
}

static size_t translator_threads() {
    const char *raw =
        std::getenv("MEETING_TRANSLATOR_THREADS");

    if (!raw || !*raw)
        return 2;

    char *end = nullptr;

    const long value =
        std::strtol(raw, &end, 10);

    if (
        end == raw
        || !end
        || *end != '\0'
        || value < 1
        || value > 8
    ) {
        throw std::runtime_error(
            "MEETING_TRANSLATOR_THREADS "
            "must be an integer in [1,8]"
        );
    }

    return static_cast<size_t>(value);
}

static std::unique_ptr<Engine>
load_engine(const std::string &dir) {
    auto e = std::make_unique<Engine>();

    if (!e->sp.Load(dir + "/spiece.model").ok()) {
        throw std::runtime_error(
            "cannot load spiece.model from " + dir
        );
    }

    ctranslate2::models::ModelLoader loader(dir);

    loader.device =
        ctranslate2::Device::CPU;

    loader.compute_type =
        ctranslate2::ComputeType::INT8_FLOAT32;

    loader.num_replicas_per_device = 1;

    ctranslate2::ReplicaPoolConfig pool;

    pool.num_threads_per_replica =
        translator_threads();

    pool.max_queued_batches = 2;

    e->translator =
        std::make_unique<ctranslate2::Translator>(
            loader,
            pool
        );

    return e;
}

static bool recognized_mode(
    const std::string &mode
) {
    return (
        mode == "beam1"
        || mode == "beam2"
        || mode == "greedy"
        || mode == "final"
        || mode == "live"
        || mode == "qa"
        || mode == "retry"
        || mode == "1"
        || mode == "2"
    );
}

static size_t beam_for_mode(
    const std::string &mode
) {
    if (
        mode == "beam2"
        || mode == "qa"
        || mode == "retry"
        || mode == "2"
    ) {
        return 2;
    }

    return 1;
}

static std::string ltrim(
    std::string value
) {
    size_t i = 0;

    while (
        i < value.size()
        && std::isspace(
            static_cast<unsigned char>(
                value[i]
            )
        )
    ) {
        ++i;
    }

    return value.substr(i);
}

static std::string strip_prefix(
    std::string value,
    const std::string &prefix
) {
    value = ltrim(value);

    if (value.rfind(prefix, 0) == 0) {
        value.erase(0, prefix.size());
        value = ltrim(value);
    }

    return value;
}

static std::string translate(
    Engine &engine,
    const std::string &direction,
    const std::string &raw,
    size_t beam_size
) {
    std::string source;
    std::string target_prefix;

    if (direction == "vi-en") {
        source = "vi: " + raw;
        target_prefix = "en:";
    } else if (direction == "en-vi") {
        source = "en: " + raw;
        target_prefix = "vi:";
    } else {
        throw std::runtime_error(
            "unsupported-direction"
        );
    }

    std::vector<std::string> pieces;

    if (
        !engine.sp.Encode(
            source,
            &pieces
        ).ok()
    ) {
        throw std::runtime_error(
            "SentencePiece encode failed"
        );
    }

    // EnViT5/T5 source contract:
    // append EOS exactly once.
    if (
        pieces.empty()
        || pieces.back() != "</s>"
    ) {
        pieces.push_back("</s>");
    }

    ctranslate2::TranslationOptions options;

    options.beam_size = beam_size;
    options.num_hypotheses = 1;
    options.return_scores = false;

    options.max_input_length = 512;
    options.max_decoding_length = 512;

    const std::vector<
        std::vector<std::string>
    > batch{pieces};

    auto results =
        engine.translator->translate_batch(
            batch,
            options
        );

    if (results.empty())
        return "";

    std::vector<std::string> output_pieces;

    for (
        const auto &piece :
        results[0].output()
    ) {
        if (
            piece == "</s>"
            || piece == "<pad>"
            || piece == "<s>"
        ) {
            continue;
        }

        output_pieces.push_back(piece);
    }

    std::string output;

    if (
        !engine.sp.Decode(
            output_pieces,
            &output
        ).ok()
    ) {
        throw std::runtime_error(
            "SentencePiece decode failed"
        );
    }

    // Smoke gate proves this model can emit
    // literal target prefixes ("en:" / "vi:").
    // Never expose those control prefixes to UI.
    output = strip_prefix(
        output,
        target_prefix
    );

    return one_line(output);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr
            << "usage: meeting-translator "
            << "ENVIT5_MODEL\n";
        return 2;
    }

    try {
        auto engine =
            load_engine(argv[1]);

        std::cout
            << "READY\tcpu\tint8_float32\n"
            << std::flush;

        std::string line;

        while (
            std::getline(
                std::cin,
                line
            )
        ) {
            if (line == "QUIT")
                break;

            const auto first_tab =
                line.find('\t');

            if (
                first_tab
                == std::string::npos
            ) {
                std::cout
                    << "ERR\tbad-request\n"
                    << std::flush;
                continue;
            }

            const std::string direction =
                line.substr(
                    0,
                    first_tab
                );

            std::string remainder =
                line.substr(
                    first_tab + 1
                );

            std::string mode = "beam1";
            std::string text = remainder;

            const auto second_tab =
                remainder.find('\t');

            if (
                second_tab
                != std::string::npos
            ) {
                const std::string candidate =
                    remainder.substr(
                        0,
                        second_tab
                    );

                if (
                    recognized_mode(
                        candidate
                    )
                ) {
                    mode = candidate;

                    text =
                        remainder.substr(
                            second_tab + 1
                        );
                }
            }

            try {
                const std::string output =
                    translate(
                        *engine,
                        direction,
                        text,
                        beam_for_mode(mode)
                    );

                if (output.empty()) {
                    std::cout
                        << "ERR\tempty-output\n"
                        << std::flush;
                } else {
                    std::cout
                        << "OK\t"
                        << output
                        << "\n"
                        << std::flush;
                }
            } catch (
                const std::exception &e
            ) {
                std::cout
                    << "ERR\t"
                    << one_line(e.what())
                    << "\n"
                    << std::flush;
            }
        }
    } catch (
        const std::exception &e
    ) {
        std::cerr
            << one_line(e.what())
            << "\n";

        return 3;
    }

    return 0;
}
