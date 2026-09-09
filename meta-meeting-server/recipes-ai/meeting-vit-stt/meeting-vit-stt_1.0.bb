SUMMARY = "Native VIT-STT bilingual runtime for HL Meet"
DESCRIPTION = "Source-rebuilt ARM64 stt-http/stt-cli plus pinned Zipformer VI and Parakeet EN assets"
LICENSE = "CLOSED"

SRC_URI = " \
    file://meeting-vit-stt.py \
    file://meeting-vit-stt-readiness.py \
    file://meeting-vit-stt.service \
    file://runtime.toml \
    file://models.local.json \
    file://assets.lock.json \
    file://vit-stt-bin-orin-aarch64.tar.zst;unpack=0 \
    file://vit-stt-meeting-models.tar.zst;unpack=0 \
"

S = "${WORKDIR}"
DEPENDS += "zstd-native"
inherit systemd

SYSTEMD_SERVICE:${PN} = "meeting-vit-stt.service"
SYSTEMD_AUTO_ENABLE:${PN} = "enable"

do_install() {
    install -d ${D}/opt/meeting/vit-stt
    tar --use-compress-program=unzstd \
        --no-same-owner --no-same-permissions \
        -xf ${WORKDIR}/vit-stt-bin-orin-aarch64.tar.zst \
        -C ${D}/opt/meeting/vit-stt

    install -d ${D}/opt/meeting/vit-stt/models
    tar --use-compress-program=unzstd \
        --no-same-owner --no-same-permissions \
        -xf ${WORKDIR}/vit-stt-meeting-models.tar.zst \
        -C ${D}/opt/meeting/vit-stt/models

    # Accuracy/release pin: verify the exact production payload, not a historical
    # outer tar byte layout. Generated archives may be rebuilt deterministically;
    # the executable/model bytes below are the release invariant.
    echo "b9d3796fd01ddec41fa0245a34526dd78f77a8b2fd91bb625428bfe2ae836d6d  ${D}/opt/meeting/vit-stt/bin/stt-http" | sha256sum -c - \
        || bbfatal "stt-http checksum mismatch"
    echo "ce2cdcc9917347dcdfb5abfc5083842839251716260e1fcfa56c8e832ed7c38c  ${D}/opt/meeting/vit-stt/bin/stt-cli" | sha256sum -c - \
        || bbfatal "stt-cli checksum mismatch"

    VIM=${D}/opt/meeting/vit-stt/models/sherpa-onnx-zipformer-vi-int8-2025-04-20
    ENM=${D}/opt/meeting/vit-stt/models/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming

    echo "b3abdef7a660fea7faf5e076b3c7613b0fc98406707103784d018189bb522124  $VIM/encoder-epoch-12-avg-8.int8.onnx" | sha256sum -c - || bbfatal "VI encoder mismatch"
    echo "d1d27cca84c824a8acf5ce6edf0f2c0880cfe295d2e69b95134de1707e1d9998  $VIM/decoder-epoch-12-avg-8.onnx" | sha256sum -c - || bbfatal "VI decoder mismatch"
    echo "38ec49e1c18e4feb0cad4de13e25c83a866cf56f4a66f22e8ff579d591a69a46  $VIM/joiner-epoch-12-avg-8.int8.onnx" | sha256sum -c - || bbfatal "VI joiner mismatch"
    echo "f536d03c2e95ebd2930cf0abec88e823bd17d3c1933da7ae6a82db3b80605e15  $VIM/tokens.txt" | sha256sum -c - || bbfatal "VI tokens mismatch"
    echo "289dbb44527c13c419ae3a4d8ce6a349f01a97f8777e69934a77e3692d2f10db  $VIM/bpe.model" | sha256sum -c - || bbfatal "VI BPE mismatch"

    echo "6716910b7a0833997fec7a410494c995d70124001a0e9b66d6370d6aced577e0  $ENM/encoder.int8.onnx" | sha256sum -c - || bbfatal "EN encoder mismatch"
    echo "a5e223392c90e75f8144cdb5eb95af7625db389e39edef2bd1a9c872b3298fe6  $ENM/decoder.int8.onnx" | sha256sum -c - || bbfatal "EN decoder mismatch"
    echo "869f43f7d24595c55581ad3bf249a935fb8a71389fbdaa7504b9f46f93140f8a  $ENM/joiner.int8.onnx" | sha256sum -c - || bbfatal "EN joiner mismatch"
    echo "dc0b4584ab2e4ddbf888425c076c61b736e7356a015250db7d307e6f1a8188ff  $ENM/tokens.txt" | sha256sum -c - || bbfatal "EN tokens mismatch"

    # meeting-server owns continuous VAD/segmentation. Do not ship historical VAD.
    rm -rf ${D}/opt/meeting/vit-stt/models/vad

    install -d ${D}/opt/meeting/vit-stt/config
    install -m 0644 ${WORKDIR}/runtime.toml ${D}/opt/meeting/vit-stt/config/runtime.toml
    install -m 0644 ${WORKDIR}/models.local.json ${D}/opt/meeting/vit-stt/config/models.local.json

    install -d ${D}/opt/meeting/vit-stt/baselines/phase0
    install -m 0644 ${WORKDIR}/assets.lock.json ${D}/opt/meeting/vit-stt/baselines/phase0/assets.lock.json

    install -d ${D}${bindir}
    install -m 0755 ${WORKDIR}/meeting-vit-stt.py ${D}${bindir}/meeting-vit-stt
    install -m 0755 ${WORKDIR}/meeting-vit-stt-readiness.py ${D}${bindir}/meeting-vit-stt-readiness

    chmod 0755 ${D}/opt/meeting/vit-stt/bin/stt-http ${D}/opt/meeting/vit-stt/bin/stt-cli
    find ${D}/opt/meeting/vit-stt/models -type f -exec chmod 0644 {} \;
    chown -R 0:0 ${D}/opt/meeting/vit-stt
    chown 0:0 ${D}${bindir}/meeting-vit-stt ${D}${bindir}/meeting-vit-stt-readiness

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/meeting-vit-stt.service ${D}${systemd_system_unitdir}/meeting-vit-stt.service
}

FILES:${PN} += " \
    /opt/meeting/vit-stt \
    ${bindir}/meeting-vit-stt \
    ${bindir}/meeting-vit-stt-readiness \
    ${systemd_system_unitdir}/meeting-vit-stt.service \
"

RDEPENDS:${PN} += " \
    libstdc++ \
    libgcc \
    ffmpeg \
    ca-certificates \
    python3-core \
    python3-json \
    python3-netclient \
"
