SUMMARY = "HL Meet EnViT5 VI-EN translation model"
DESCRIPTION = "Pinned VietAI EnViT5 CTranslate2 INT8_FLOAT32 model"
LICENSE = "CLOSED"

SRC_URI = " \
    file://envit5-ct2.tar \
    file://envit5-runtime.sha256 \
"

S = "${WORKDIR}"

inherit allarch

do_install() {
    # The outer tar contains non-runtime provenance metadata. Certify the
    # actual inference payload instead of coupling the build to container bytes.
    (
        cd ${WORKDIR}/envit5
        sha256sum -c ${WORKDIR}/envit5-runtime.sha256
    ) || bbfatal "EnViT5 runtime payload checksum mismatch"

    install -d \
        ${D}/opt/meeting/models/translation/envit5

    cp -R --no-preserve=ownership \
        ${WORKDIR}/envit5/. \
        ${D}/opt/meeting/models/translation/envit5/

    chown -R root:root \
        ${D}/opt/meeting/models/translation/envit5
}

FILES:${PN} = "/opt/meeting/models/translation/envit5"
