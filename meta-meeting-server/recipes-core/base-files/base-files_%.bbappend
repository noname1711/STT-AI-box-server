do_install:append() {
    echo "meeting-server" > ${D}${sysconfdir}/hostname
}
