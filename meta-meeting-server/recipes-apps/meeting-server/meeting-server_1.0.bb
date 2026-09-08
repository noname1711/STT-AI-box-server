SUMMARY = "HL Meet V20 transition VIT-only STT API"
LICENSE = "CLOSED"

SRC_URI = " \
    file://server.py \
    file://meeting-server.service \
    file://meeting-server.env \
"

S = "${WORKDIR}"

inherit systemd

SYSTEMD_SERVICE:${PN} = "meeting-server.service"
SYSTEMD_AUTO_ENABLE = "enable"

do_install() {
    install -d ${D}/opt/meeting/server
    install -m 0755 ${WORKDIR}/server.py ${D}/opt/meeting/server/server.py

    install -d ${D}${systemd_system_unitdir}
    install -m 0644 ${WORKDIR}/meeting-server.service \
        ${D}${systemd_system_unitdir}/meeting-server.service

    install -d ${D}${sysconfdir}/meeting-server
    install -m 0644 ${WORKDIR}/meeting-server.env \
        ${D}${sysconfdir}/meeting-server/meeting-server.env
}

FILES:${PN} += " \
    /opt/meeting/server \
    ${sysconfdir}/meeting-server/meeting-server.env \
"

CONFFILES:${PN} += "${sysconfdir}/meeting-server/meeting-server.env"

# v19.4 keeps the locked VIT baseline untouched. Translation is an isolated
# post-ASR lane with MADLAD INT8 primary and OPUS rollback compatibility.
# Whisper, Turbo and ASR arbitration packages remain excluded.
RDEPENDS:${PN} += " \
    python3-core \
    python3-difflib \
    python3-aiohttp \
    python3-asyncio \
    python3-json \
    python3-logging \
    python3-misc \
    meeting-vit-stt \
"
