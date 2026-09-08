cmake_minimum_required(VERSION 3.16)

project(meeting_translator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ctranslate2 CONFIG REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(
    SPM
    REQUIRED
    IMPORTED_TARGET
    sentencepiece
)

add_executable(
    meeting-translator
    meeting-translator.cpp
)

target_link_libraries(
    meeting-translator
    PRIVATE
    CTranslate2::ctranslate2
    PkgConfig::SPM
)

install(
    TARGETS meeting-translator
    RUNTIME DESTINATION bin
)
