# Vietnamese CAPU candidate benchmark

Generated at: `2026-05-27T11:40:41+0700`

## Scope

This benchmark compares candidate Vietnamese capitalization and punctuation postprocessors against the current locked ViBERT CAPU production outputs. The runtime-facing package is `leakless/vibert-capu`, a consolidated copy of the accepted ViBERT CAPU weights with `base_model/` included. The score is therefore a replacement-risk benchmark, not an independent linguistic gold benchmark.

## Candidate Sources

- `leakless/vibert-capu`: current production package. This is the consolidated ViBERT CAPU package used by `vit-stt`; it includes the fine-tuned CAPU files plus `base_model/` so the runtime does not need to assemble `dragonSwing/vibert-capu` and `FPTAI/vibert-base-cased` separately. Source: https://huggingface.co/leakless/vibert-capu
- `dragonSwing/vibert-capu`: upstream source model for the accepted CAPU behavior. Hugging Face reports ViBERT Seq2Labels, OSCAR-2109 Vietnamese training data, `cc-by-sa-4.0`, and held-out label F1 scores: upper `0.89`, complex-upper `0.88`, period `0.82`, comma `0.71`, colon `0.64`, question `0.78`. Source: https://huggingface.co/dragonSwing/vibert-capu
- `dragonSwing/xlm-roberta-capu`: same task family and training data, but XLM-RoBERTa based. Hugging Face reports slightly better comma/colon metrics than ViBERT: comma `0.72`, colon `0.67`, with similar period/question scores. Source: https://huggingface.co/dragonSwing/xlm-roberta-capu
- `welcomyou/vibert-capu-onnx`: public ONNX packaging of `dragonSwing/vibert-capu`, including FP32 and INT8 ONNX graphs. Hugging Face metadata lists `dragonSwing/vibert-capu` as base model, `cc-by-sa-4.0`, and ONNX Runtime as the library. Source: https://huggingface.co/welcomyou/vibert-capu-onnx
- `tourmii/vietnamese-punc-cap-denorm-v1`: mBART text2text model for punctuation, capitalization, and Vietnamese number/date denormalization. It has no comparable published benchmark in the model card and no explicit license metadata in the Hugging Face API response observed during this pass. Source: https://huggingface.co/tourmii/vietnamese-punc-cap-denorm-v1

## Methodology

- Runtime: macOS arm64, CPU only, Python `3.11`, PyTorch `2.12.0`, Transformers `4.30.2`.
- Inputs: 4 locked Phase 0 snippets from `baselines/phase0/capu_snippets.jsonl` plus 8 audio-derived transcript prefixes from the previous CAPU long-form benchmark (`weanxinviec`, `thoitiet936`, and `bomman`, up to 4m prefixes).
- Reference output: current production ViBERT CAPU output from locked baselines and prior audio-derived Python CAPU runs.
- Iterations: 2 measured inference runs per case after model initialization and first-call timing.
- Metrics:
  - `Exact`: byte-for-byte match against current production CAPU output.
  - `Punctuation exact`: exact match of only `.,:?!` punctuation sequence.
  - `Mean char edit rate`: Levenshtein distance divided by reference length.
  - Latency: mean wall-clock CPU inference time across all case runs.
- Limitation: this is not a human-labeled linguistic benchmark. It answers whether a candidate is a safe replacement for the existing product behavior. A full quality benchmark would require a manually reviewed Vietnamese ASR transcript set with accepted punctuation/capitalization references.

## Summary

| Candidate | Size MB | Init ms | First ms | Mean ms | Exact | Punctuation exact | Mean char edit rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vibert-capu-current | 462.6 | 1422.6 | 37.0 | 190.8 | 12/12 | 12/12 | 0.000 |
| xlm-roberta-capu | 1113.4 | 4673.4 | 41.8 | 169.5 | 2/12 | 3/12 | 0.025 |
| tourmii-vietnamese-punc-cap-denorm-v1 | 1588.9 | 2839.1 | 464.0 | 8768.7 | 1/12 | 2/12 | 0.167 |

Additional ONNX packaging check:

| Candidate | Model file | Size | Init ms | First ms | Mean ms on `model_card_example` | Locked snippet parity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `welcomyou/vibert-capu-onnx` | `vibert-capu.onnx` | 438.2 MB | 531.8 | 125.9 | 130.2 | 0/4 |
| `welcomyou/vibert-capu-onnx` | `vibert-capu.int8.onnx` | 110.3 MB | 389.4 | 95.1 | 94.0 | 0/4 |

## Findings

### Current ViBERT CAPU

`vibert-capu-current` remains the reference behavior. It matched all locked snippets and long transcript prefixes exactly. The measured local mean was `190.8 ms`, with long-form cases naturally dominating the mean. It also has the smallest disk footprint of the tested neural candidates: `462.6 MB` measured by the script, `441 MB` by `du -sh`.

### XLM-RoBERTa CAPU

`xlm-roberta-capu` is the only plausible near-term replacement candidate. It was slightly faster on this fixture set (`169.5 ms` mean vs `190.8 ms`) but much larger (`1113.4 MB`, about 2.4x the current model) and had meaningful behavior drift:

- Exact match: `2/12`.
- Punctuation exact match: `3/12`.
- Mean char edit rate: `0.025`.
- It changed punctuation in short and conversational cases, for example `Định nghĩa: Thế nào là ăn mặc đẹp?` instead of `Định nghĩa thế nào là ăn mặc đẹp?`.
- It under-capitalized some proper/formal phrases relative to the locked baseline, for example `kỳ họp thứ nhất Quốc hội khóa mười sáu` instead of `Kỳ họp thứ nhất Quốc hội khóa Mười sáu`.
- On conversational `bomman` prefixes, it removed several sentence/question boundaries and produced a flatter punctuation style.

Interpretation: XLM-R may be useful as an optional experimental mode, but it is not a safe default replacement. Its public model-card metrics are marginally better on comma/colon, but local behavior is not better for our locked product outputs.

### Tourmii mBART Denormalization Model

`tourmii-vietnamese-punc-cap-denorm-v1` is not suitable as a CAPU replacement in the current runtime:

- Exact match: `1/12`.
- Punctuation exact match: `2/12`.
- Mean char edit rate: `0.167`.
- Mean CPU inference was `8768.7 ms`, about 46x slower than the current CAPU path on this benchmark.
- It intentionally changes text normalization: `mười sáu` became `XVI`, `mười lăm triệu` became `15 triệu`, and temperature ranges became digits.
- It produced malformed output on the `bomman/4m` case: `"Chào buổi, tối là "việc", "đểu", "viên", "được"!.`
- The model has no explicit license metadata in the observed Hugging Face API result, so redistribution/integration is not cleared.

Interpretation: this model may be interesting for a separate transcript-formatting or denormalization feature, but it should not back `postprocess_mode=capu`.

### Welcomyou ViBERT CAPU ONNX

`welcomyou/vibert-capu-onnx` is not a new quality model; it is a public ONNX packaging of `dragonSwing/vibert-capu`. It is useful to evaluate because it could avoid maintaining our own export if it preserved parity.

Result: as tested through the Rust ONNX spike, it does not preserve parity.

- FP32 ONNX locked snippet parity: `0/4`.
- INT8 ONNX locked snippet parity: `0/4`.
- FP32 latency on the `model_card_example` snippet: `130.2 ms` mean, slower than the Python worker `91.5 ms` in the same run.
- INT8 latency on the same snippet: `94.0 ms` mean, roughly equal to the Python worker `91.8 ms`, with smaller model size (`110.3 MB`).
- FP32 output example: `Theo đó thủ tướng dự kiến tiếp, bộ trưởng nông nghiệp mỹ Tom Wilsack...`
- INT8 output example: `Theo đó thủ tướng dự kiến tiếp, bộ trưởng nông nghiệp, mỹ tom, wilsack...`

The public ONNX graphs have a slightly different input signature from our internal export (`token_type_ids` is required), so the Rust spike was updated to feed zero `token_type_ids` when present. The graphs execute successfully after reshaping the repo into our local export bundle format, but their outputs are not equivalent to the locked Python CAPU behavior.

Interpretation: `welcomyou/vibert-capu-onnx` is not safe to adopt as the default ONNX artifact. The INT8 graph is interesting for size and latency, but exact CAPU parity fails badly enough that it should remain a reference point only. The existing local ONNX/ORT-format export path from the May 25 spike is still the better packaging candidate because it preserved locked snippet parity.

## Case-level output drift

### vibert-capu-current

| Case | Exact | Punct exact | Edit rate | Mean ms | Output |
| --- | :---: | :---: | ---: | ---: | --- |
| snippet/probe | yes | yes | 0.000 | 34.4 | Định nghĩa thế nào là ăn mặc đẹp? |
| snippet/long_vi_segment | yes | yes | 0.000 | 53.2 | Xin kính chào quý vị khán giả. Thưa quý vị, chỉ ít phút nữa thì Kỳ họp thứ nhất Quốc hội khóa Mười sáu sẽ được khai mạc trọng thể. |
| snippet/legacy_expected_test_transcript | yes | yes | 0.000 | 27.1 | Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia. |
| snippet/model_card_example | yes | yes | 0.000 | 56.2 | Theo đó, Thủ tướng dự kiến tiếp Bộ trưởng Nông nghiệp Mỹ Tom Wilsack, Bộ trưởng Thương mại Mỹ Gina Raimondo, Bộ trưởng Tài chính Janet Yellen, gặp gỡ Thượng nghị sĩ Patrick Leah... |
| asset/weanxinviec/30s | yes | yes | 0.000 | 62.2 | Dạ, ờ, định nghĩa thế nào là ăn mặc đẹp? Là anh vẫn em rồi, sếp à. Em định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô đó không phải là mình mình. Em cần phải tiếp tục nói vào ... |
| asset/weanxinviec/2m | yes | yes | 0.000 | 69.1 | Dạ, ờ, định nghĩa thế nào là ăn mặc đẹp? Là anh vẫn em rồi, sếp à. Em định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô đó không phải là mình mình. Em cần phải tiếp tục nói vào ... |
| asset/thoitiet936/30s | yes | yes | 0.000 | 80.0 | Quý vị đang theo dõi bản tin Dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/thoitiet936/2m | yes | yes | 0.000 | 250.0 | Quý vị đang theo dõi bản tin Dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/thoitiet936/4m | yes | yes | 0.000 | 397.3 | Quý vị đang theo dõi bản tin Dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/bomman/30s | yes | yes | 0.000 | 122.6 | Chào buổi tối nha. I mới ra trường lương mười lăm triệu. Ok không anh ơi? Ui ngon vãi em. Mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương... |
| asset/bomman/2m | yes | yes | 0.000 | 407.0 | Chào buổi tối nha. I mới ra trường lương mười lăm triệu. Ok không anh ơi? Ui ngon vãi em. Mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương... |
| asset/bomman/4m | yes | yes | 0.000 | 729.9 | Chào buổi tối nha. I mới ra trường lương mười lăm triệu. Ok không anh ơi? Ui ngon vãi em. Mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương... |

### xlm-roberta-capu

| Case | Exact | Punct exact | Edit rate | Mean ms | Output |
| --- | :---: | :---: | ---: | ---: | --- |
| snippet/probe | no | no | 0.061 | 41.1 | Định nghĩa: Thế nào là ăn mặc đẹp? |
| snippet/long_vi_segment | no | yes | 0.015 | 32.7 | Xin kính chào quý vị khán giả. Thưa quý vị, chỉ ít phút nữa thì kỳ họp thứ nhất Quốc hội khóa mười sáu sẽ được khai mạc trọng thể. |
| snippet/legacy_expected_test_transcript | yes | yes | 0.000 | 24.8 | Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia. |
| snippet/model_card_example | yes | yes | 0.000 | 55.4 | Theo đó, Thủ tướng dự kiến tiếp Bộ trưởng Nông nghiệp Mỹ Tom Wilsack, Bộ trưởng Thương mại Mỹ Gina Raimondo, Bộ trưởng Tài chính Janet Yellen, gặp gỡ Thượng nghị sĩ Patrick Leah... |
| asset/weanxinviec/30s | no | no | 0.029 | 63.0 | Dạ ờ, định nghĩa thế nào là ăn mặc đẹp? Là anh vẫn em rồi sếp à, em định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô đó không phải là mình mình. Em cần phải tiếp tục nói vào Qu... |
| asset/weanxinviec/2m | no | no | 0.029 | 46.1 | Dạ ờ, định nghĩa thế nào là ăn mặc đẹp? Là anh vẫn em rồi sếp à, em định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô đó không phải là mình mình. Em cần phải tiếp tục nói vào Qu... |
| asset/thoitiet936/30s | no | no | 0.014 | 78.9 | Quý vị đang theo dõi bản tin dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/thoitiet936/2m | no | no | 0.009 | 226.6 | Quý vị đang theo dõi bản tin dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông. Cần đ... |
| asset/thoitiet936/4m | no | no | 0.006 | 375.2 | Quý vị đang theo dõi bản tin dự báo thời tiết của báo Nhân Dân. Thưa quý vị, hôm nay khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông. Cần đ... |
| asset/bomman/30s | no | no | 0.055 | 120.7 | Chào buổi tối nha, I mới ra trường lương mười lăm triệu ok không anh ơi. Ui ngon vãi em mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương a... |
| asset/bomman/2m | no | no | 0.043 | 344.1 | Chào buổi tối nha, I mới ra trường lương mười lăm triệu ok không anh ơi. Ui ngon vãi em mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương a... |
| asset/bomman/4m | no | no | 0.034 | 625.1 | Chào buổi tối nha, I mới ra trường lương mười lăm triệu ok không anh ơi. Ui ngon vãi em mười lăm triệu mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương a... |

### tourmii-vietnamese-punc-cap-denorm-v1

| Case | Exact | Punct exact | Edit rate | Mean ms | Output |
| --- | :---: | :---: | ---: | ---: | --- |
| snippet/probe | no | no | 0.030 | 508.4 | Định nghĩa thế nào là ăn mặc đẹp?. |
| snippet/long_vi_segment | no | no | 0.123 | 1155.6 | Xin kính chào quý vị khán giả, thưa quý vị, chỉ ít phút nữa thì Kỳ họp thứ nhất, Quốc hội XVI sẽ được khai mạc trọng thể. |
| snippet/legacy_expected_test_transcript | yes | yes | 0.000 | 557.1 | Rồi cũng hỗ trợ cho, lâu lâu cũng cho gạo cho này kia. |
| snippet/model_card_example | no | yes | 0.024 | 1740.8 | Theo đó, Thủ tướng dự kiến tiếp Bộ trưởng Nông nghiệp Mỹ Tom Wilsack, Bộ trưởng Thương mại Mỹ Ned Raimondo, Bộ trưởng Tài chính Janet Yellen, gặp gỡ thượng nghị sĩ Patrick Leahy... |
| asset/weanxinviec/30s | no | no | 0.078 | 2117.7 | "Dạ ờ định nghĩa thế nào là ăn mặc đẹp là anh vẫn em rồi, sếp à?". "Định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô, đó không phải là mình mình, em cần phải tiếp tục nói vào Q... |
| asset/weanxinviec/2m | no | no | 0.077 | 2155.0 | "Dạ ờ định nghĩa thế nào là ăn mặc đẹp là anh vẫn em rồi, sếp à?". "Định nghĩa ăn mặc đẹp có nghĩa là nó mình nhìn vô, đó không phải là mình mình, em cần phải tiếp tục nói vào Q... |
| asset/thoitiet936/30s | no | no | 0.028 | 2927.0 | Quý vị đang theo dõi bản tin dự báo thời tiết của Báo Nhân Dân thưa quý vị, hôm nay, khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/thoitiet936/2m | no | no | 0.105 | 12054.1 | Quý vị đang theo dõi bản tin dự báo thời tiết của Báo Nhân Dân thưa quý vị, hôm nay, khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/thoitiet936/4m | no | no | 0.287 | 26038.1 | Quý vị đang theo dõi bản tin dự báo thời tiết của Báo Nhân Dân thưa quý vị, hôm nay, khu vực Bắc Bộ nhiều nơi có mưa, vùng núi và trung du có nơi mưa vừa, mưa to kèm dông, cần đ... |
| asset/bomman/30s | no | no | 0.103 | 4570.7 | Chào buổi tối nha,i mới ra trường lương 15 triệu, ok không anh ơi, ui ngon vãi em 15 triệu, mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương anh được có ... |
| asset/bomman/2m | no | no | 0.163 | 25615.0 | Chào buổi tối nha,i mới ra trường lương 15 triệu, ok không anh ơi, Ui ngon vãi em 15 triệu, mới ra trường anh thấy là ngon thôi chứ mới ra trường ngày xưa anh lương anh được có ... |
| asset/bomman/4m | no | no | 0.990 | 25784.9 | "Chào buổi, tối là "việc", "đểu", "viên", "được"!. |

## Recommendation

Keep `leakless/vibert-capu` as the default CAPU package and keep the local model id `vibert-capu`.

Do not replace it with `dragonSwing/xlm-roberta-capu` right now. XLM-R is the best alternative found, but the local benchmark shows output drift in punctuation and capitalization while increasing model footprint substantially. If we want to explore it further, add it as an opt-in model ID and run a human review on a larger Vietnamese meeting/call-center transcript set.

Do not use `tourmii/vietnamese-punc-cap-denorm-v1` for `postprocess_mode=capu`. It is too slow on CPU, changes transcript semantics through denormalization, fails long conversational input, and has unresolved licensing metadata.

Do not replace our internal CAPU export with `welcomyou/vibert-capu-onnx`. Its INT8 file is attractive operationally, but both FP32 and INT8 failed locked snippet parity in this harness.

Recommended next steps:

1. Keep current `vibert-capu` as the production default.
2. Preserve the benchmark harness in `scripts/benchmark_capu_candidates.py` and raw results in `results.json`.
3. If model quality work continues, build a human-reviewed CAPU evaluation set with at least 100 Vietnamese ASR transcript segments across news, meetings, casual speech, and noisy call-center text.
4. Evaluate XLM-R only as an experimental candidate after that larger set exists.
5. Treat denormalization as a separate product decision, not a CAPU replacement.
