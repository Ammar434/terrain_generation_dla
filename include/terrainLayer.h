#pragma once
#include "simulationConfig.h"
#include <QDateTime>
#include <QDebug>
#include <QImage>
#include <QString>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
class TerrainLayers {
public:
  explicit TerrainLayers(int width, int height, const SimulationConfig &config);
  ~TerrainLayers();

  // Process and save layers
  void processLayers(const std::vector<float> &baseHeightMap);

  // Save functions
  bool saveLayerImages() const;
  bool saveLayerImage(size_t layerIndex) const;
  bool saveCombinedImage() const;

  // Access functions
  const std::vector<float> &getLayer(size_t index) const;
  std::vector<float> getCombinedLayers() const;
  size_t getNumLayers() const { return layers.size(); }
  int getWidth() const { return width; }
  int getHeight() const { return height; }

private:
  void applyGaussianSmoothing(const std::vector<float> &input,
                              std::vector<float> &output, float radius) const;
  QImage createImageFromHeightMap(const std::vector<float> &heightMap) const;
  QString getDefaultLayerName(size_t index) const;

  int width;
  int height;
  SimulationConfig config;
  std::vector<std::vector<float>> layers;

  // CUDA resources
  float *d_input = nullptr;
  float *d_output = nullptr;
};