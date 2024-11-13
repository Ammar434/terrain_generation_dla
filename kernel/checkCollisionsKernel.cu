#include <cuda_runtime.h>
#include <particle.h>
#include "constant.h"

__device__ float calculateSquaredDistance(const Particle &p1, const Particle &p2)
{
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

__global__ void checkCollisionsKernel(Particle *particles,
                                      int numParticles,
                                      float particleRadius)
{
    int xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int yIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIdx >= numParticles || yIdx >= numParticles || xIdx >= yIdx)
    {
        return;
    }

    Particle &particleX = particles[xIdx];
    Particle &particleY = particles[yIdx];

    // Skip if both particles are active or both are frozen
    if (particleX.isActive == particleY.isActive)
    {
        return;
    }

    // Determine which particle is frozen/active
    int frozenIdx;
    const Particle &frozenParticle = !particleX.isActive ? particleX : particleY;
    Particle &activeParticle = particleX.isActive ? particleX : particleY;
    frozenIdx = !particleX.isActive ? xIdx : yIdx;

    const float freezeRadiusSquared = 4.0f * particleRadius * particleRadius;
    float squaredDistance = calculateSquaredDistance(frozenParticle, activeParticle);

    if (squaredDistance <= freezeRadiusSquared)
    {
        activeParticle.collidedParticleIdx = frozenIdx;
    }
}

void launchCheckCollisionsKernel(Particle *particles,
                                 int numParticles,
                                 float particleRadius)
{

    constexpr int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim(
        (numParticles + blockDim.x - 1) / blockDim.x,
        (numParticles + blockDim.y - 1) / blockDim.y);

    checkCollisionsKernel<<<gridDim, blockDim>>>(particles,
                                                 numParticles,
                                                 particleRadius);

    cudaDeviceSynchronize();
}