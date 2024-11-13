// simulation.h
#pragma once
#include "particle.h"
#include "simulationConfig.h"
#include <QObject>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

class CudaResources {
public:
  CudaResources() = default;
  ~CudaResources();

  void allocate(size_t numParticles);
  void deallocate();

  Particle *d_particles = nullptr;
  bool *d_allFrozen = nullptr;
  curandState *d_states = nullptr;
};

class Simulation : public QObject {
  Q_OBJECT

public:
  explicit Simulation(const SimulationConfig &config,
                      QObject *parent = nullptr);
  ~Simulation();

  // Core simulation methods
  void runSimulation();

  // Accessors
  const std::vector<float> &getHeightMap() const { return heightMap; }
  const std::vector<Particle> &getParticles() const { return particles; }
  bool isComplete() const { return allFrozen; }
  int getCurrentStep() const { return currentStep; }

signals:
  void simulationStepCompleted(int step);
  void simulationCompleted();
  void errorOccurred(const QString &error);

private:
  // Initialization methods
  void initializeParticles();
  void initializeCuda();
  void updateHeightMap();

  // CUDA kernel wrappers
  void runCudaStep();

  // Utility methods
  float getRandomFloat(float min, float max);

  // Configuration
  SimulationConfig config;

  // Simulation state
  std::vector<Particle> particles;
  std::vector<float> heightMap;
  int currentStep = 0;
  bool allFrozen = false;

  // CUDA resources
  std::unique_ptr<CudaResources> cudaResources;

  // RNG
  std::mt19937 rng{std::random_device{}()};
};
