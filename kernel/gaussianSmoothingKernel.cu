#include <cuda_runtime.h>
#include <particle.h>
#include <curand_kernel.h>

__global__ void gaussianSmoothingKernel(
    float *input,
    float *output,
    int width,
    int height,
    float radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    output[idx] = input[idx]; // Default to input value

    // Only process valid particles
    if (input[idx] >= 0)
    {
        float sum = 0.0f;
        float weightSum = 0.0f;

        // Define window size for local search
        int windowRadius = (int)(3 * radius); // 3-sigma rule for Gaussian

        // Compute bounds for the local window
        int startX = max(0, x - windowRadius);
        int endX = min(width - 1, x + windowRadius);
        int startY = max(0, y - windowRadius);
        int endY = min(height - 1, y + windowRadius);

        // Process local window instead of entire grid
        for (int ny = startY; ny <= endY; ny++)
        {
            for (int nx = startX; nx <= endX; nx++)
            {
                int nIdx = ny * width + nx;

                float dx = float(nx - x);
                float dy = float(ny - y);
                float dist = sqrtf(dx * dx + dy * dy);

                // Calculate Gaussian weight
                float weight = expf(-(dist * dist) / (2.0f * radius * radius));
                sum += input[nIdx] * weight;
                weightSum += weight;
            }
        }

        // Update value if we found neighboring particles
        if (weightSum > 0.0f)
        {
            output[idx] = sum / weightSum;
        }
    }
}

void launchGaussianSmoothingKernel(
    float *input,
    float *output,
    int width,
    int height,
    float radius)
{
    constexpr int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    // Calculate grid dimensions
    dim3 gridDim(
        (width + blockDim.x - 1) / blockDim.x,
        (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    gaussianSmoothingKernel<<<gridDim, blockDim>>>(
        input,
        output,
        width,
        height,
        radius);
    cudaDeviceSynchronize();
}