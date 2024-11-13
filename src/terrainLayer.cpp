#include "terrainLayer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// External CUDA function declaration
extern void launchGaussianSmoothingKernel(float *input, float *output,
                                          int width, int height, float radius);

TerrainLayers::TerrainLayers(int width, int height,
                             const SimulationConfig &config)
    : width(width), height(height), config(config), d_input(nullptr),
      d_output(nullptr) {
  // Allocate CUDA memory
  size_t mapSize = width * height * sizeof(float);
  cudaError_t error = cudaMalloc(&d_input, mapSize);
  if (error != cudaSuccess) {
    throw std::runtime_error("Failed to allocate input CUDA memory");
  }

  error = cudaMalloc(&d_output, mapSize);
  if (error != cudaSuccess) {
    cudaFree(d_input);
    throw std::runtime_error("Failed to allocate output CUDA memory");
  }
}

TerrainLayers::~TerrainLayers() {
  cudaFree(d_input);
  cudaFree(d_output);
}

void TerrainLayers::processLayers(const std::vector<float> &baseHeightMap) {
  if (baseHeightMap.size() != static_cast<size_t>(width * height)) {
    throw std::invalid_argument("Invalid heightmap dimensions");
  }

  // Clear existing layers
  layers.clear();
  layers.reserve(config.numLayers);

  // First layer is the original DLA pattern
  layers.push_back(baseHeightMap);

  // Create smoothed versions with exponentially increasing radius
  std::vector<float> smoothedLayer(width * height);
  float radius = 2.0f; // Start with small radius

  for (int i = 1; i < config.numLayers; ++i) {
    // Take original DLA and apply increasing smoothing
    smoothedLayer = baseHeightMap;
    applyGaussianSmoothing(smoothedLayer, smoothedLayer, radius);
    layers.push_back(smoothedLayer);

    // Double the radius for each layer (exponential increase)
    radius *= 2.0f; // This gives radii of 2, 4, 8, 16, 32, 64...
    qDebug() << "Created layer" << i << "with radius" << radius;
  }

  saveLayerImages(); // Save for visualization
  saveCombinedImage();
}

void TerrainLayers::applyGaussianSmoothing(const std::vector<float> &input,
                                           std::vector<float> &output,
                                           float radius) const {
  size_t mapSize = width * height * sizeof(float);

  // Copy input to device
  cudaMemcpy(d_input, input.data(), mapSize, cudaMemcpyHostToDevice);

  // Apply smoothing
  launchGaussianSmoothingKernel(d_input, d_output, width, height, radius);

  // Copy result back to host
  cudaMemcpy(output.data(), d_output, mapSize, cudaMemcpyDeviceToHost);
}

const std::vector<float> &TerrainLayers::getLayer(size_t index) const {
  if (index >= layers.size()) {
    throw std::out_of_range("Layer index out of range");
  }
  return layers[index];
}

std::vector<float> TerrainLayers::getCombinedLayers() const {
  if (layers.empty()) {
    return std::vector<float>();
  }

  const size_t pixelCount = width * height;
  std::vector<float> result(pixelCount, 0.0f);

  // First, normalize each layer and add them with decreasing weights
  float weight = 1.0f;

  for (size_t i = 0; i < layers.size(); i++) {
    // Normalize current layer to [0,1] range
    std::vector<float> normalizedLayer = layers[i];
    float minVal =
        *std::min_element(normalizedLayer.begin(), normalizedLayer.end());
    float maxVal =
        *std::max_element(normalizedLayer.begin(), normalizedLayer.end());
    float range = maxVal - minVal;

    if (range > 0) {
      for (size_t j = 0; j < pixelCount; j++) {
        normalizedLayer[j] = (normalizedLayer[j] - minVal) / range;
      }
    }

    // Add normalized layer with weight
    for (size_t j = 0; j < pixelCount; j++) {
      int intensity = static_cast<int>(normalizedLayer[j] * 255.0f);
      intensity = std::clamp(intensity, 0, 255);
      result[j] += normalizedLayer[j] * weight;
    }

    weight /= 0.15f;
  }

  // Add contrast enhancement
  for (size_t j = 0; j < pixelCount; j++) {
    result[j] = std::pow(result[j], 1.2f);
  }

  return result;
}

QString TerrainLayers::getDefaultLayerName(size_t index) const {
  QString timestamp =
      (QDateTime::currentDateTime()).toString("yyyy-MM-dd_hh-mm-ss");
  return QString("%1/layer_%2_%3.png")
      .arg(config.outputDirectory)
      .arg(index, 2, 10, QChar('0'))
      .arg(timestamp);
}

QImage TerrainLayers::createImageFromHeightMap(
    const std::vector<float> &heightMap) const {
  if (heightMap.empty())
    return QImage();

  QImage image(width, height, QImage::Format_RGB32);

  float minHeight = *std::min_element(heightMap.begin(), heightMap.end());
  float maxHeight = *std::max_element(heightMap.begin(), heightMap.end());
  float range = maxHeight - minHeight;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int index = y * width + x;

      float normalizedHeight = (heightMap[index] - minHeight) / range;

      // Convert to intensity [0,255]
      int intensity = static_cast<int>(normalizedHeight * 255.0f);
      intensity = std::clamp(intensity, 20, 255);

      image.setPixel(x, y, qRgb(intensity, intensity, intensity));
    }
  }

  return image;
}

bool TerrainLayers::saveLayerImage(size_t layerIndex) const {
  if (layerIndex >= layers.size())
    return false;

  QString filename = getDefaultLayerName(layerIndex);
  QImage image = createImageFromHeightMap(layers[layerIndex]);

  if (image.save(filename)) {
    qDebug() << "Saved layer" << layerIndex << "to" << filename;
    return true;
  } else {
    qWarning() << "Failed to save layer" << layerIndex << "to" << filename;
    return false;
  }
}

bool TerrainLayers::saveLayerImages() const {
  bool allSuccess = true;

  for (size_t i = 0; i < layers.size(); ++i) {
    if (!saveLayerImage(i)) {
      allSuccess = false;
    }
  }

  return allSuccess;
}

bool TerrainLayers::saveCombinedImage() const {
  if (layers.empty())
    return false;

  std::vector<float> combined = getCombinedLayers();
  QString timestamp =
      QDateTime::currentDateTime().toString("yyyy-MM-dd_hh-mm-ss");
  QString filename =
      QString("%1/combined_%2.png").arg(config.outputDirectory).arg(timestamp);

  QImage image = createImageFromHeightMap(combined);

  if (image.save(filename)) {
    qDebug() << "Saved combined layers to" << filename;
    return true;
  } else {
    qWarning() << "Failed to save combined layers to" << filename;
    return false;
  }
}
