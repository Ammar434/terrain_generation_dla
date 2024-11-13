
// simulation.cpp
#include "simulation.h"
#include <QDebug>
#include <stdexcept>

// CUDA kernel declarations
extern void launchSetupRandomStatesKernel(curandState *states,
                                          unsigned long seed, int size);
extern void launchMoveParticlesKernel(Particle *particles, int numParticles,
                                      float moveRadius, int width, int height,
                                      curandState *states);
extern void launchFreezeParticlesKernel(Particle *particles, int numParticles,
                                        float particleRadius, int currentStep,
                                        bool *allFrozen);
extern void launchCheckCollisionsKernel(Particle *particles, int numParticles,
                                        float particleRadius);
extern void launchGaussianSmoothingKernel(float *input, float *output,
                                          int width, int height, float radius);

CudaResources::~CudaResources() { deallocate(); }

void CudaResources::allocate(size_t numParticles) {
  cudaMalloc(&d_particles, numParticles * sizeof(Particle));
  cudaMalloc(&d_allFrozen, sizeof(bool));
  cudaMalloc(&d_states, numParticles * sizeof(curandState));
}

void CudaResources::deallocate() {
  cudaFree(d_particles);
  cudaFree(d_allFrozen);
  cudaFree(d_states);
  d_particles = nullptr;
  d_allFrozen = nullptr;
  d_states = nullptr;
}

Simulation::Simulation(const SimulationConfig &config, QObject *parent)
    : QObject(parent), config(config), particles(config.numParticles),
      heightMap(config.width * config.height) {
  try {
    initializeParticles();
    initializeCuda();
  } catch (const std::exception &e) {
    emit errorOccurred(QString("Initialization error: %1").arg(e.what()));
  }
}

Simulation::~Simulation() = default;

void Simulation::initializeParticles() {

  // Distribution for particle positions
  std::uniform_real_distribution<float> xDist(
      config.spawnMargin, config.width - config.spawnMargin);
  std::uniform_real_distribution<float> yDist(
      config.spawnMargin, config.height - config.spawnMargin);

  // Track how many particles we've frozen
  int remainingFrozenParticles = config.initialFrozenParticles;

  if (remainingFrozenParticles == 1) {
    particles[0].x = config.width / 2.0f; // Center position
    particles[0].y = config.height / 2.0f;
    particles[0].oldX = particles[0].x;
    particles[0].oldY = particles[0].y;
    particles[0].isActive = false;
    particles[0].frozenAtStep = 0;
    remainingFrozenParticles--;
  }

  for (size_t i = 1; i < particles.size(); ++i) {
    particles[i].x = xDist(rng);
    particles[i].y = yDist(rng);
    particles[i].oldX = particles[i].x;
    particles[i].oldY = particles[i].y;

    if (remainingFrozenParticles > 0) {
      particles[i].isActive = false;
      particles[i].frozenAtStep = 0;
      remainingFrozenParticles--;
    } else {
      particles[i].isActive = true;
      particles[i].frozenAtStep = -1;
    }
  }
}
void Simulation::initializeCuda() {
  cudaResources = std::make_unique<CudaResources>();
  cudaResources->allocate(particles.size());
  launchSetupRandomStatesKernel(cudaResources->d_states, rng(),
                                particles.size());
}

void Simulation::runSimulation() {
  try {
    cudaMemcpy(cudaResources->d_particles, particles.data(),
               particles.size() * sizeof(Particle), cudaMemcpyHostToDevice);

    while (!allFrozen && currentStep < 100000) {
      runCudaStep();

      if (currentStep % 10000 == 0) {
        bool frozenState = true;
        cudaMemcpy(cudaResources->d_allFrozen, &frozenState, sizeof(bool),
                   cudaMemcpyHostToDevice);
        emit simulationStepCompleted(currentStep);
      }

      currentStep++;
    }

    cudaMemcpy(particles.data(), cudaResources->d_particles,
               particles.size() * sizeof(Particle), cudaMemcpyDeviceToHost);

    updateHeightMap();
    emit simulationCompleted();
  } catch (const std::exception &e) {
    emit errorOccurred(QString("Simulation error: %1").arg(e.what()));
  }
}

void Simulation::runCudaStep() {
  launchMoveParticlesKernel(cudaResources->d_particles, particles.size(),
                            config.moveRadius, config.width, config.height,
                            cudaResources->d_states);

  launchCheckCollisionsKernel(cudaResources->d_particles, particles.size(),
                              config.particleRadius);

  launchFreezeParticlesKernel(cudaResources->d_particles, particles.size(),
                              config.particleRadius, currentStep,
                              cudaResources->d_allFrozen);

  if (currentStep % 10000 == 0) {
    cudaMemcpy(&allFrozen, cudaResources->d_allFrozen, sizeof(bool),
               cudaMemcpyDeviceToHost);
  }
}

// TODO : Correct heightmap elevation for more than 1 frozen particle
void Simulation::updateHeightMap() {
  std::fill(heightMap.begin(), heightMap.end(), 0.0f);

  const float maxDistance =
      std::sqrt(config.width * config.width + config.height * config.height);

  for (const auto &particle : particles) {
    if (!particle.isActive) {
      float dx = particle.x - particles[0].x;
      float dy = particle.y - particles[0].y;
      float distance = std::sqrt(dx * dx + dy * dy);
      float elevation = 255.0f * (1.0f - distance / maxDistance);

      int x = static_cast<int>(particle.x);
      int y = static_cast<int>(particle.y);

      if (x >= 0 && x < config.width && y >= 0 && y < config.height) {
        heightMap[y * config.width + x] = elevation;
      }
    }
  }
}

float Simulation::getRandomFloat(float min, float max) {
  std::uniform_real_distribution<float> dist(min, max);
  return dist(rng);
}
