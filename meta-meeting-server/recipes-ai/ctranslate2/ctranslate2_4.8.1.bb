SUMMARY = "Optimized Transformer inference runtime"
HOMEPAGE = "https://github.com/OpenNMT/CTranslate2"
LICENSE = "CLOSED"

SRC_URI = "gitsm://github.com/OpenNMT/CTranslate2.git;protocol=https;branch=master"
SRCREV = "a93c2bfb8af9c702f42f355f17a29952289d1d1f"

S = "${WORKDIR}/git"

# Keep the existing Ruy CPU backend for runtime rollback, and compile the
# CUDA backend for Jetson Orin Nano. meta-tegra's cuda.bbclass supplies the
# CUDA 12.6 cross-toolchain/sysroot integration on the scarthgap BSP.
inherit cmake cuda

EXTRA_OECMAKE += " \
    -DBUILD_CLI=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DWITH_MKL=OFF \
    -DWITH_DNNL=OFF \
    -DWITH_ACCELERATE=OFF \
    -DWITH_OPENBLAS=OFF \
    -DWITH_RUY=ON \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=OFF \
    -DCUDA_ARCH_LIST=8.7 \
    -DCUDA_DYNAMIC_LOADING=OFF \
    -DWITH_TENSOR_PARALLEL=OFF \
    -DOPENMP_RUNTIME=NONE \
    -DENABLE_CPU_DISPATCH=OFF \
"
