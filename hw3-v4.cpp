#include "utils.h"

__constant__  float F[25];

template<int TILE_WIDTH, int R>
__global__ void convolution_v1(
    float* A, // [M,N]
    float* B, // [M,N]
    int M,
    int N
) {

    __shared__ float sA[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + (blockDim.y * blockIdx.y),
        col = threadIdx.x + (blockDim.x * blockIdx.x);

    // load non-halo cells phase
    sA[threadIdx.y][threadIdx.x] = (row < M && col < N) ? A[row * N + col] : 0.0f;
    __syncthreads();

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int i = -R; i < R + 1; i++) {
            for (int j = -R; j < R + 1; j++) {
                // we're trying to compute A[row + i][col + j] *  F[r + i][r + j]
                // it's not halo if: (blockIdx.y) * blockDim.y < row + i < (blockIdx.y + 1) * blockDim.y (within the block)
                if (
                    row + i > (int)(blockIdx.y * blockDim.y) - 1 &&
                    row + i < (int)((blockIdx.y + 1) * blockDim.y) &&
                    col + j > (int)(blockIdx.x * blockDim.x) - 1 &&
                    col + j < (int)((blockIdx.x + 1) * blockDim.x)
                ) {
                    // can safely access shared memory
                    sum += sA[threadIdx.y + i][threadIdx.x + j] * F[(R + i) * (2*R + 1) + (R + j)];
                } else {
                    int in_row = (row + i);
                    int in_col = (col + j);
                    float val_A = (in_row > -1 && in_row < M && in_col > -1 && in_col < N ) ? A[in_row * N + in_col] : 0.0f;
                    sum += F[(R + i) * (2*R + 1) + (R + j)] * val_A;
                }
            }
        }

        B[row * N + col] = sum;
    }
}

float block_size_experiments(
    int M,
    int N,
    dim3 grid_size,
    dim3 block_size
)
{
    // Memory is allocated and freed on both RAM and GPU automatically
    // when creating HeterogenousMatrix class.
    HeterogeneousMatrix A(M, N);
    HeterogeneousMatrix C(M, N);

    // Generate input matrices:
    initialize_matrix_hw3(A);
    A.to_device();

    float filter[25] = {
        0.01f, 0.02f, 0.04f, 0.02f, 0.01f,
        0.02f, 0.05f, 0.08f, 0.05f, 0.02f,
        0.04f, 0.08f, 0.12f, 0.08f, 0.04f,
        0.02f, 0.05f, 0.08f, 0.05f, 0.02f,
        0.01f, 0.02f, 0.04f, 0.02f, 0.01f
    };

    cuda_check(cudaMemcpyToSymbol(F, filter, sizeof(filter)));

    // Timing done based on the following tutorial https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    switch (block_size.x) {
        case 8:
            convolution_v1<8, 2><<<grid_size, block_size>>>(
                A.d_data, C.d_data, M, N);
            break;
        case 16:
            convolution_v1<16, 2><<<grid_size, block_size>>>(
                A.d_data, C.d_data, M, N);
            break;
        case 32:
            convolution_v1<32, 2><<<grid_size, block_size>>>(
                A.d_data, C.d_data, M, N);
            break;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    C.to_host();

    std::cout << "C(0,0) = " << C.h_data[0] << '\n'
          << "C(0,1) = " << C.h_data[1] << '\n'
          << "C(1,0) = " << C.h_data[C.n_cols] << '\n'
          << "C(1,1) = " << C.h_data[C.n_cols + 1] << '\n'
          << "C(5000, 5000) = " << C.h_data[5000 * C.n_cols + 5000] << '\n'
          << "C(10000, 10000) = " << C.h_data[10000 * C.n_cols + 10000] << '\n'
          << "C(15000, 15000) = "  << C.h_data[15000 * C.n_cols + 15000] << '\n'
          << "C(19999,19999) = " << C.h_data[19999 * C.n_cols + 19999] << '\n';


    return milliseconds;
}


int main() {
    int M = 20000;
    int N = 20000;

    dim3 block_sizes[] = {
        dim3(8, 8, 1),
        dim3(16, 16, 1),
        dim3(32, 32, 1)
    };

    for (auto block_size : block_sizes) {

        dim3 grid_size(
            (N + block_size.x - 1) / block_size.x,
            (M + block_size.y - 1) / block_size.y
        );

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }

}
