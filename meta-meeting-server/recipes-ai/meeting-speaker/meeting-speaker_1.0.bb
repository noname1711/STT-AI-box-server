SUMMARY = "HL Meet V20 offline speaker diarization"
DESCRIPTION = "Offline speaker diarization for HL Meet using sherpa-onnx, Pyannote segmentation and 3D-Speaker embedding."
HOMEPAGE = "https://github.com/k2-fsa/sherpa-onnx"

LICENSE = "Apache-2.0 & MIT"

LIC_FILES_CHKSUM = " \
    file://Apache-2.0.txt;md5=3b83ef96387f14655fc854ddc3c6bd57 \
    file://PYANNOTE_LICENSE.txt;md5=490418c2dd02008d8859a903b921cbcf \
"

SRC_URI = " \
    file://meeting-speaker \
    file://meeting-speaker-id.cpp \
    file://meeting-speaker-seg-reid.cpp \
    file://sherpa-onnx-diarization-abi-v1.13.2.h \
    file://sherpa-onnx-c-api-v1.13.2.h \
    file://sherpa-onnx-v1.13.2-linux-aarch64-static-lib.tar.bz2;subdir=sherpa-static \
    file://meeting-speaker-v20-daemon.py \
    file://meeting-speaker-v20.service \
    file://speaker-segmentation.onnx \
    file://speaker-embedding.onnx \
    file://ASSETS.sha256 \
    file://MODEL_PROVENANCE.md \
    file://Apache-2.0.txt \
    file://PYANNOTE_LICENSE.txt \
"

S = "${WORKDIR}"

COMPATIBLE_HOST = "aarch64.*-linux"

inherit systemd

SYSTEMD_SERVICE:${PN} = "meeting-speaker-v20.service"
SYSTEMD_AUTO_ENABLE = "enable"

RDEPENDS:${PN} += " python3-core python3-asyncio python3-logging "


do_compile() {
    set -eu

    SHERPA_ROOT="${WORKDIR}/sherpa-static/sherpa-onnx-v1.13.2-linux-aarch64-static-lib"
    SHERPA_LIB="${SHERPA_ROOT}/lib"

    test -f "${SHERPA_LIB}/libsherpa-onnx-c-api.a"
    test -f "${SHERPA_LIB}/libsherpa-onnx-core.a"
    test -f "${SHERPA_LIB}/libonnxruntime.a"

    ${CXX} ${CXXFLAGS} \
        -std=c++17 \
        -O3 \
        -DNDEBUG \
        -I${WORKDIR} \
        ${WORKDIR}/meeting-speaker-id.cpp \
        ${LDFLAGS} \
        -Wl,--start-group \
        ${SHERPA_LIB}/*.a \
        -Wl,--end-group \
        -pthread \
        -ldl \
        -lm \
        -lrt \
        -o ${WORKDIR}/meeting-speaker-id

    test -s ${WORKDIR}/meeting-speaker-id

    ${CXX} ${CXXFLAGS} \
        -std=c++17 \
        -O3 \
        -DNDEBUG \
        -I${WORKDIR} \
        ${WORKDIR}/meeting-speaker-seg-reid.cpp \
        ${LDFLAGS} \
        -Wl,--start-group \
        ${SHERPA_LIB}/*.a \
        -Wl,--end-group \
        -pthread \
        -ldl \
        -lm \
        -lrt \
        -o ${WORKDIR}/meeting-speaker-seg-reid

    test -s ${WORKDIR}/meeting-speaker-seg-reid
}

do_install() {
    install -d ${D}/opt/meeting/speaker/bin
    install -d ${D}/opt/meeting/speaker/models
    install -d ${D}/opt/meeting/speaker/share
    install -d ${D}${libexecdir}
    install -d ${D}${systemd_system_unitdir}

    install -m 0755 \
        ${WORKDIR}/meeting-speaker \
        ${D}/opt/meeting/speaker/bin/meeting-speaker

    install -m 0755 \
        ${WORKDIR}/meeting-speaker-id \
        ${D}/opt/meeting/speaker/bin/meeting-speaker-id

    install -m 0755 \
        ${WORKDIR}/meeting-speaker-seg-reid \
        ${D}/opt/meeting/speaker/bin/meeting-speaker-seg-reid

    install -m 0644 \
        ${WORKDIR}/speaker-segmentation.onnx \
        ${D}/opt/meeting/speaker/models/speaker-segmentation.onnx

    install -m 0644 \
        ${WORKDIR}/speaker-embedding.onnx \
        ${D}/opt/meeting/speaker/models/speaker-embedding.onnx

    install -m 0644 \
        ${WORKDIR}/MODEL_PROVENANCE.md \
        ${D}/opt/meeting/speaker/share/MODEL_PROVENANCE.md

    install -m 0755 \
        ${WORKDIR}/meeting-speaker-v20-daemon.py \
        ${D}${libexecdir}/meeting-speaker-v20-daemon.py

    install -m 0644 \
        ${WORKDIR}/meeting-speaker-v20.service \
        ${D}${systemd_system_unitdir}/meeting-speaker-v20.service

    (
        cd ${D}/opt/meeting/speaker
        sha256sum \
            bin/meeting-speaker \
            bin/meeting-speaker-id \
            bin/meeting-speaker-seg-reid \
            models/speaker-segmentation.onnx \
            models/speaker-embedding.onnx \
            > share/ASSETS.sha256
    )
}

FILES:${PN} += " \
    /opt/meeting/speaker \
    ${libexecdir}/meeting-speaker-v20-daemon.py \
    ${systemd_system_unitdir}/meeting-speaker-v20.service \
"
