cmake_minimum_required(VERSION 3.16)
project(terrain_generation_dla LANGUAGES CXX CUDA)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 14)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets OpenGL)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets OpenGL)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(PROJECT_SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/mainwindow.ui
    src/simulation.cpp
    src/heightmapwidget.cpp
    src/terrainLayer.cpp

    include/mainwindow.h
    include/simulation.h
    include/heightmapwidget.h
    include/particle.h
    include/constant.h
    include/terrainLayer.h
    include/simulationConfig.h

    kernel/setupRandomStatesKernel.cu
    kernel/moveParticlesKernel.cu
    kernel/freezeParticlesKernel.cu
    kernel/checkCollisionsKernel.cu
    kernel/gaussianSmoothingKernel.cu
)

add_executable(terrain_generation_dla ${PROJECT_SOURCES})

target_include_directories(terrain_generation_dla PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/kernel
)

target_link_libraries(terrain_generation_dla PRIVATE 
    Qt${QT_VERSION_MAJOR}::Widgets
    Qt${QT_VERSION_MAJOR}::OpenGL
    ${OPENGL_LIBRARIES}
    CUDA::cudart
)

# Only call qt_finalize_executable for Qt6
if(${QT_VERSION_MAJOR} EQUAL 6)
    qt_finalize_executable(terrain_generation_dla)
endif()