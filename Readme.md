# Terrain Generation using DLA (Diffusion Limited Aggregation)
<div align="center">
<a href="https://isocpp.org/"><img src="https://img.shields.io/badge/Made%20with-C%2B%2B-00599C?style=for-the-badge&logo=c%2B%2B" alt="Made with C++"></a>
<a href="https://www.qt.io/"><img src="https://img.shields.io/badge/Qt-41CD52?style=for-the-badge&logo=qt&logoColor=white" alt="Made with Qt"></a>
<a href="https://developer.nvidia.com/cuda-zone"><img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA Accelerated"></a>
<br>
<img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat" alt="Contributions welcome">
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
</div>

## 📋 Overview


https://github.com/user-attachments/assets/55113b6f-6255-4ba8-bd25-12a1e17716f7



This project implements a GPU-accelerated terrain generation system using Diffusion Limited Aggregation (DLA) with real-time OpenGL visualization. The implementation combines DLA-based mountain formation with multi-scale Gaussian filtering to create realistic terrain features.

## 🌄 Features
- CUDA-accelerated DLA particle simulation
- Real-time OpenGL terrain visualization
- Multiple terrain visualization themes (Desert, Arctic, Alien, Volcanic, etc.)
- Interactive camera controls with mouse and keyboard
- Multi-scale Gaussian terrain smoothing
- Support for both real-time generation and pre-built heightmaps

## 🛠️ Technologies
- C++17
- CUDA Toolkit
- Qt6/Qt5 for GUI and OpenGL integration
- OpenGL for 3D rendering
- CMake build system

## 🔬 Technical Methodology

### 1. Terrain Generation Pipeline
The terrain generation consists of three main stages:
1. Initial noise generation using Diffusion Limited Aggregation (DLA)
2. Multi-scale Gaussian filtering using CUDA
3. Real-time OpenGL rendering with theme-based visualization

### 2. DLA Implementation (Noise Generation)
The DLA process uses CUDA for parallel particle simulation:

```cpp
// Particle structure
class Particle {
    float x, y;              // Current position
    float oldX, oldY;        // Previous position
    float elevation;         // Height contribution
    bool isActive;          // Particle state
    int frozenAtStep;       // Freezing timestamp
    int collidedParticleIdx; // Collision tracking
    int iteration;          // Current iteration
};
```

The DLA algorithm:
1. Initialize particles in random positions around the boundary
2. Particles perform random walks
3. When colliding with frozen particles, they freeze and contribute to elevation
4. Process continues until desired mountain formation is achieved

Key CUDA kernels:
```cpp
__global__ void moveParticlesKernel(
    Particle* particles,
    curandState* states,
    int particleCount
);

__global__ void checkCollisionsKernel(
    Particle* particles,
    float* heightMap,
    float particleRadius
);

__global__ void freezeParticlesKernel(
    Particle* particles,
    float* heightMap,
    int width,
    int height
);
```
![layer_01_2024-11-13_04-24-20](https://github.com/user-attachments/assets/88593f5e-8a88-4fb9-ad07-bb4987f1dd68)

![layer_01_2024-11-13_02-29-52](https://github.com/user-attachments/assets/dde86d3d-ff77-4467-8d94-7386638a5564)

### 3. Multi-scale Filtering
Multiple Gaussian filters are applied with exponentially increasing radii:

```cpp
__global__ void gaussianSmoothingKernel(
    float* input,
    float* output,
    int width,
    int height,
    float radius
);
```
The filtering process creates a hierarchy of terrain features at different scales.
<div align="center">

| Layer | Description | Visualization |
|:-----:|-------------|:-------------:|
| Layer 1 | Initial DLA Output | ![layer_01](https://github.com/user-attachments/assets/f8612cbb-7f72-41fd-a3fa-f6604ee1bd52) |
| Layer 2 | First Smoothing Pass | ![layer_02](https://github.com/user-attachments/assets/0097bd20-67bf-4605-9fee-02a1687242c3) |
| Layer 3 | Medium-Scale Features | ![layer_03](https://github.com/user-attachments/assets/a1dec2f4-1937-4dd4-8c76-aacbbcf47257) |
| Layer 4 | Large-Scale Features | ![layer_04](https://github.com/user-attachments/assets/8bc9505a-4072-45cb-8dce-3dd6e2321baa) |
| Layer 5 | Major Landforms | ![layer_05](https://github.com/user-attachments/assets/49ca7420-3049-4a49-8bb7-d950b1182dc9) |

### 4. Final Combined Result
After combining all layers with appropriate weights:

<img src="https://github.com/user-attachments/assets/ac5793d0-8b28-4f9b-ab7f-dce5c1ea7a17" alt="Combined Result" width="600"/>

</div>

## 📋 Prerequisites
- NVIDIA GPU with CUDA support (Optional - required only for new heightmap generation)
- CUDA Toolkit 11.0+ (Optional)
- Qt6 or Qt5
- CMake 3.16+
- C++17 compatible compiler

## 🚀 Building from Source
### Dependencies Installation
```bash
# Install Qt
sudo apt-get install qt6-base-dev qt6-opengl-dev

# Install CUDA Toolkit (Optional - for new heightmap generation)
# Download from: https://developer.nvidia.com/cuda-downloads
```

### Building
```bash
git clone https://github.com/yourusername/terrain-generation-dla.git
cd terrain-generation-dla
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## 💻 Usage
```bash
./terrain_generation_dla
```

### Controls
- Left Mouse Button: Rotate camera
- Mouse Wheel: Zoom in/out
- 1-7 Keys: Switch terrain themes
- Space: Reset view

## 🔧 Configuration
The main configuration is in `simulationConfig.h`. For quick testing, you can use a pre-built heightmap:

```cpp
struct SimulationConfig {
    // Basic parameters
    int width = 1080;
    int height = 1080;
    int numParticles = 15000;
    float particleRadius = 2.0f;
    float moveRadius = 1.0f;
    float spawnMargin = 10.0f;

    // Terrain generation
    float initialRadius = 1.5f;    // Initial smoothing radius
    int numLayers = 6;             // Number of detail layers
    int initialFrozenParticles = 1;

    // Heightmap options
    bool buildYourOwnHeightmap = false;  // Set true to generate new heightmap
    QString heightMapPath = "./heightmap/five_frozen/combined_2024-11-13_01-20-11.png";
    QString outputDirectory = "heightmap/one_frozen";
};
```

## 📚 References
- Josh's Channel. ["Better Mountain Generators That Aren't Perlin Noise or Erosion"](https://www.youtube.com/watch?v=gsJHzBTPG0Y)
- Guillaume Belz. ["Qt OpenGL - Générer un terrain"](http://guillaume.belz.free.fr/doku.php?id=qt_opengl_-_generer_un_terrain)
- NVIDIA CUDA Programming Guide
- OpenGL Programming Guide

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
