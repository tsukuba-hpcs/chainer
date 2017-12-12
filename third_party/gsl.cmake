cmake_minimum_required(VERSION 2.8.2)

project(gsl-download NONE)

include(ExternalProject)
ExternalProject_Add(gsl-lite
        GIT_REPOSITORY    https://github.com/martinmoene/gsl-lite
        GIT_TAG           master
        SOURCE_DIR        "${CMAKE_BINARY_DIR}/gsl-lite"
        BINARY_DIR        ""
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      ""
        )
