SUMMARY = "HL Meet EnViT5 VI-EN translation model"
DESCRIPTION = "Pinned VietAI EnViT5 CTranslate2 INT8_FLOAT32 model"
LICENSE = "CLOSED"

SRC_URI = "file://envit5-ct2.tar"
SRC_URI[sha256sum] = "e5ddd932173cca21fd181eab4b2888955ff291ce0cbefcdca1fdb5333bd01588"

S = "${WORKDIR}"

inherit allarch

do_install() {
    install -d         ${D}/opt/meeting/models/translation/envit5

    cp -R --no-preserve=ownership         ${WORKDIR}/envit5/.         ${D}/opt/meeting/models/translation/envit5/

    chown -R root:root         ${D}/opt/meeting/models/translation/envit5
}

FILES:${PN} = "/opt/meeting/models/translation/envit5"
