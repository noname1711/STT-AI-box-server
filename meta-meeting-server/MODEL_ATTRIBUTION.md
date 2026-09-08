# Model attribution and redistribution notes — HL Meet AI Box v19.7

This file records what can be established from the current layer plus older project attribution notes.

It is **not legal advice**. Before commercial redistribution, independently verify the license terms for every model weight shipped in the image.

## 1. Whisper Small

Current model path:

```text
/opt/meeting/models/whisper/ggml-small.bin
```

Older project notes identify:

```text
runtime project: whisper.cpp
code license:    MIT
model family:    OpenAI Whisper
```

The current `meeting-models` recipe packages the model from local `models.tar.zst`; it does not itself record the upstream URL or revision.

For release records, keep the original fetch provenance, SHA-256, and upstream model/license notice.

## 2. Whisper Large-v3-Turbo Q5 arbiter

Current packaged model:

```text
/opt/meeting/models/whisper/ggml-large-v3-turbo-q5_0.bin
```

Current verified SHA-256:

```text
394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2
```

The current `meeting-model-arbiter` recipe declares MIT and installs:

```text
/usr/share/licenses/meeting-model-arbiter/LICENSE.openai-whisper
/usr/share/licenses/meeting-model-arbiter/MODEL-NOTICE.txt
```

Preserve those files in redistributed images.

## 3. Translation models

### v19.7 production: VietAI EnViT5 INT8

Runtime directory:

```text
/opt/meeting/models/translation/envit5
```

Pinned upstream model:

```text
repository: VietAI/envit5-translation
revision:   840bc88104d5a4277af740eaedb024df8c3093e7
source pytorch_model.bin SHA-256: eef48b3eee23aae577e965ce8da5b2e9dcadfc4d08a85e2a302ef4b929fb613e
```

The v19.7 deployment uses CTranslate2 4.8.1 on CPU with INT8_FLOAT32. The
converted production assets are pinned by SHA-256 in the model-lab gate and in
`PROVENANCE.txt` inside the model pack. EnViT5 uses source prefixes `en:` and
`vi:` and the native SentencePiece adapter appends the T5 `</s>` source EOS
exactly once to reproduce the validated tokenizer contract.

The upstream model metadata identifies an OpenRAIL license. Preserve the exact
pinned upstream license/model-card terms and independently review redistribution
and commercial-use obligations before release. This layer keeps `LICENSE =
"CLOSED"` for the local package rather than re-labelling model-weight rights.

Historical MADLAD/OPUS translation packs are rollback/test artifacts and are not
shipped in the v19.7 production EnViT5 pack.

## 4. Native VIT-STT Vietnamese model

```text
id:        vit_stt_vi_v2
family:    Gipformer 65M RNNT
path:      /opt/meeting/vit-stt/models/stt/gipformer-65M-rnnt
format:    INT8 ONNX transducer
```

Current `assets.lock.json` records hashes for the encoder, decoder, joiner, tokens, and BPE model.

The current v17 metadata supplied here does **not** record the upstream repository, copyright owner, or model-weight license.

That provenance must be added before commercial redistribution.

## 5. Native VIT-STT English model

```text
id:        vit_stt_en_v2
family:    NeMo Parakeet 0.6B
path:      /opt/meeting/vit-stt/models/stt/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming
format:    INT8 ONNX NeMo transducer
```

Current `assets.lock.json` records hashes for encoder, decoder, joiner, and tokens.

The current v17 metadata supplied here does **not** establish the exact upstream revision or redistribution license for the model weights.

Record and verify that provenance before commercial release.

## 6. Runtime libraries

Current AI stack includes recipes for:

```text
whisper-cpp
CTranslate2
SentencePiece
```

Third-party runtime notices should be maintained separately from model-weight attribution.

## Release checklist

For every release, archive:

```text
model filename
model SHA-256
upstream project/repository
exact revision/tag
license identifier
copyright/attribution text
local conversion/quantization procedure, if any
```

Do not infer model-weight redistribution rights only from the license of the inference runtime.
