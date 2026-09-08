SUMMARY = "HL Meet EnViT5 VI-EN translator"
DESCRIPTION = "FINAL post-ASR EnViT5 CTranslate2 translator"
LICENSE = "CLOSED"

SRC_URI = " \
    file://CMakeLists.txt \
    file://meeting-translator.cpp \
    file://meeting-translator-daemon.py \
    file://meeting-translator.service \
"

S = "${WORKDIR}"

inherit cmake pkgconfig systemd

DEPENDS += " \
    ctranslate2 \
    sentencepiece \
"

RDEPENDS:${PN} += " \
    meeting-translation-models \
    python3-core \
    python3-asyncio \
    python3-logging \
"

SYSTEMD_SERVICE:${PN} = \
    "meeting-translator.service"

SYSTEMD_AUTO_ENABLE = "enable"

do_install:append() {
    install -d \
        ${D}${libexecdir}

    install -m 0755 \
        ${WORKDIR}/meeting-translator-daemon.py \
        ${D}${libexecdir}/meeting-translator-daemon.py

    install -d \
        ${D}${systemd_system_unitdir}

    install -m 0644 \
        ${WORKDIR}/meeting-translator.service \
        ${D}${systemd_system_unitdir}/meeting-translator.service
}

FILES:${PN} += " \
    ${libexecdir}/meeting-translator-daemon.py \
    ${systemd_system_unitdir}/meeting-translator.service \
"
