#include <cuda_runtime.h>
#include <particle.h>
#include <curand_kernel.h>

__global__ void moveParticlesKernel(Particle *particles,
                                    int numParticles,
                                    float moveRadius,
                                    int width, int height,
                                    curandState *states)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // guard against out of bounds access and inactive particles
    auto particle = particles + idx;
    if (idx > numParticles || !particle->isActive)
    {
        return;
    }

    // Reset position if particle has been wandering too long
    // if (particle->iteration > 20000)
    // {
    //     // Respawn near the edges of the simulation
    //     float margin = width * 0.3f; // 10% margin from edges

    //     // Randomly choose which edge to spawn from
    //     if (curand_uniform(&(states[idx])) < 0.5f)
    //     {
    //         // Spawn on left or right edge
    //         particle->x = static_cast<float>(curand_uniform(&(states[idx])) < 0.5f ? margin : width - margin);
    //         particle->y = static_cast<float>(margin + curand_uniform(&(states[idx])) * (height - 2 * margin));
    //     }
    //     else
    //     {
    //         // Spawn on top or bottom edge
    //         particle->x = static_cast<float>(margin + curand_uniform(&(states[idx])) * (width - 2 * margin));
    //         particle->y = static_cast<float>(curand_uniform(&(states[idx])) < 0.5f ? margin : height - margin);
    //     }

    //     particle->oldX = particle->x;
    //     particle->oldY = particle->y;
    //     particle->iteration = 0;
    //     return;
    // }

    // generate random move
    float angle = curand_uniform(&(states[idx])) * 2 * M_PI;
    const float dx = moveRadius * __cosf(angle);
    const float dy = moveRadius * __sinf(angle);

    // save the old position
    particle->oldX = particle->x;
    particle->oldY = particle->y;

    // move the particle
    particle->x += dx;
    particle->y += dy;

    // clip the coordinates to stay within the bounds of the simulation
    particle->x = fmax(0.f, fmin(particle->x, static_cast<float>(width)));
    particle->y = fmax(0.f, fmin(particle->y, static_cast<float>(height)));

    particle->iteration++;
}

void launchMoveParticlesKernel(Particle *particles,
                               int numParticles,
                               float moveRadius,
                               int width, int height,
                               curandState *states)
{
    int blockSize = 1024;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;

    moveParticlesKernel<<<numBlocks, blockSize>>>(particles,
                                                  numParticles,
                                                  moveRadius,
                                                  width, height,
                                                  states);

    cudaDeviceSynchronize();
}