SUMMARY = "SentencePiece tokenizer library"
HOMEPAGE = "https://github.com/google/sentencepiece"
LICENSE = "CLOSED"

SRC_URI = "git://github.com/google/sentencepiece.git;protocol=https;branch=master"
SRCREV = "31646a467d2051eb904e0b45de3a73e91fe1c1e3"

S = "${WORKDIR}/git"

inherit cmake

EXTRA_OECMAKE += " \
    -DSPM_ENABLE_SHARED=ON \
    -DSPM_BUILD_TEST=OFF \
    -DSPM_ENABLE_TCMALLOC=OFF \
    -DSPM_PROTOBUF_PROVIDER=internal \
    -DSPM_USE_BUILTIN_PROTOBUF=ON \
"

FILES:${PN}:append = " ${bindir}/spm_*"
