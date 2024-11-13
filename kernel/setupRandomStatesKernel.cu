#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void setupRandomStatesKernel(curandState *states, unsigned long seed, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

void launchSetupRandomStatesKernel(curandState *states, unsigned long seed, int size)
{
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    setupRandomStatesKernel<<<numBlocks, blockSize>>>(states, seed, size);
    cudaDeviceSynchronize();
}