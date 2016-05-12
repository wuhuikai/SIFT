cmake_minimum_required (VERSION 3.1)
project (SIFT)

set (CMAKE_CXX_STANDARD 11)

add_compile_options(-Werror -Wextra -O3)

FIND_LIBRARY(OPENCV opencv_core)
FIND_LIBRARY(OPENCV_GUI opencv_highgui)
FIND_LIBRARY(OPENCV_IM opencv_imgproc)
FIND_LIBRARY(OPENCV_NON opencv_nonfree)
FIND_LIBRARY(OPENCV_FEAT opencv_features2d)

SET(EXTRA_LIBS ${OPENCV} ${OPENCV_GUI} ${OPENCV_IM} ${OPENCV_NON} ${OPENCV_FEAT})

FILE(GLOB SIFT_SRC src/*.cpp)

set(SRCS ${SIFT_SRC})
add_executable(SIFT ${SRCS})

TARGET_LINK_LIBRARIES(SIFT ${EXTRA_LIBS})

FILE(GLOB IMG imgs/*)
FILE(COPY ${IMG} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/imgs)