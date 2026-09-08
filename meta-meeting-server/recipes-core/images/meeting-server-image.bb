SUMMARY = "HL Meet V20 STT + post-ASR translation AI Box"

require recipes-demo/images/demo-image-common.inc

inherit extrausers

# Access is an explicit image profile. `lab` preserves the existing development
# SSH/password workflow; commercial builds must set HL_MEET_ACCESS_PROFILE=release.
HL_MEET_ACCESS_PROFILE ?= "lab"

# Exactly one ASR stack is installed: meeting-vit-stt.
# EnViT5 is post-ASR translation only and never participates in canonical STT.
IMAGE_INSTALL:append = " \
    meeting-server \
    meeting-vit-stt \
    meeting-translator \
    python3-core \
    python3-aiohttp \
    python3-asyncio \
    python3-json \
    python3-logging \
    python3-misc \
    openssh \
    openssh-sftp-server \
    ca-certificates \
    tzdata \
    connman \
    connman-client \
    avahi-daemon \
    avahi-utils \
    iproute2 \
    iputils \
    curl \
"

HL_MEET_LAB_PASSWORD_HASH ?= "\$6\$hlmeetv191\$UI95sQOkqZWLt3u95bonMGChsV2ZWifoDg4lh74NBEIS/Thw4.Dj6Q2UbUFUeFHvdxxVni1QBDdK1PvHV5Rnq."

EXTRA_USERS_PARAMS = " \
    useradd -m -s /bin/sh -G video hungle; \
"

python __anonymous() {
    profile = d.getVar("HL_MEET_ACCESS_PROFILE") or "lab"
    if profile == "lab":
        password_hash = d.getVar("HL_MEET_LAB_PASSWORD_HASH") or ""
        if not password_hash:
            bb.fatal("HL_MEET_LAB_PASSWORD_HASH is empty in lab profile")
        d.appendVar("IMAGE_INSTALL", " sudo")
        d.appendVar(
            "EXTRA_USERS_PARAMS",
            " usermod -p '%s' hungle;" % password_hash,
        )
    elif profile == "release":
        d.appendVar("EXTRA_USERS_PARAMS", " usermod -L hungle;")
    else:
        bb.fatal("HL_MEET_ACCESS_PROFILE must be lab or release")
}

# Keep comfortable room for native model assets and A/B rootfs flashing.
IMAGE_ROOTFS_EXTRA_SPACE = "4194304"

configure_meeting_access() {
    if [ "${HL_MEET_ACCESS_PROFILE}" = "lab" ]; then
        install -d -m 0755 ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d
        install -m 0440 /dev/null \
            ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d/hungle
        echo 'hungle ALL=(ALL) ALL' \
            > ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d/hungle
    elif [ "${HL_MEET_ACCESS_PROFILE}" = "release" ]; then
        rm -f ${IMAGE_ROOTFS}${sysconfdir}/sudoers.d/hungle
    else
        bbfatal "HL_MEET_ACCESS_PROFILE must be lab or release"
    fi
}

ROOTFS_POSTPROCESS_COMMAND += "configure_meeting_access; "
