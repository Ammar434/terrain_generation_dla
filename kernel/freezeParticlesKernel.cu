#include <cuda_runtime.h>
#include <particle.h>
#include <curand_kernel.h>
#include <stdio.h>

// Todo : Remove all auto

__device__ void surfaceCollisionPoint(float xa, float ya, // active particle
                                      float xf, float yf, // frozen particle
                                      float dx, float dy, // move vector
                                      float r,            // particle radius
                                      float *x, float *y)
{

    // link to the math: https://www.sciencedirect.com/science/article/pii/S0010465511001238#br0150
    float x2 = xa + dx, y2 = ya + dy;
    float bx = x2 - xf, by = y2 - yf;

    float theta = 2 * (bx * dx + by * dy);
    float psi = bx * bx + by * by - 4 * r * r;
    float chi = dx * dx + dy * dy;

    float alpha = (-theta - sqrt(theta * theta - 4 * chi * psi)) / (2 * chi);

    *x = x2 + alpha * dx;
    *y = y2 + alpha * dy;
}

__global__ void freezeParticlesKernel(Particle *particles,
                                      int numParticles,
                                      float particleRadius,
                                      int currentStep,
                                      bool *allFrozen)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numParticles)
    {
        return;
    }

    Particle &particle = particles[idx];

    // If particle is not active, skip it
    if (!particle.isActive)
    {
        return;
    }

    // Mark that we have at least one active particle
    *allFrozen = false;

    // If no collision detected, return
    if (particle.collidedParticleIdx == -1)
    {
        return;
    }

    Particle &frozenParticle = particles[particle.collidedParticleIdx];

    // Calculate exact collision point
    float dx = particle.x - particle.oldX;
    float dy = particle.y - particle.oldY;
    float collision_x, collision_y;

    surfaceCollisionPoint(particle.oldX, particle.oldY,
                          frozenParticle.x, frozenParticle.y,
                          dx, dy,
                          particleRadius,
                          &collision_x, &collision_y);

    // Freeze the particle
    particle.isActive = false;
    particle.frozenAtStep = currentStep;
    particle.x = collision_x;
    particle.y = collision_y;
}
void launchFreezeParticlesKernel(Particle *particles,
                                 int numParticles,
                                 float particleRadius,
                                 int currentStep,
                                 bool *allFrozen)
{
    int blockSize = 1024;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    freezeParticlesKernel<<<numBlocks, blockSize>>>(particles,
                                                    numParticles,
                                                    particleRadius,
                                                    currentStep,
                                                    allFrozen);
    // cudaDeviceSynchronize();
}