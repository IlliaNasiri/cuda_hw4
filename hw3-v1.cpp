#include "utils.h"


__global__ void convolution_v1(
    float* A, // [M,N]
    float* F, // [5,5]
    float *B, // [M,N]
    int M,
    int N,
    int r
) {

    int row = threadIdx.y + (blockIdx.y * blockDim.y);
    int col = threadIdx.x + (blockIdx.x * blockDim.x);

    // we don't do anthing if we're outside M and N:
    float sum = 0.0f;
    if (row < M && col < N) {
        for (int j = -r; j < r + 1; j++) {
            for (int i = -r; i < r + 1; i++) {
                int a_row = row + j;
                int a_col = col + i;

                float a_value = ( a_row > -1 && a_row < M && a_col > -1 && a_col < N ) ? A[ a_row * N + a_col ] : 0.0f;
                sum += a_value * F[(r + j) * (2 * r + 1) + (r + i)];

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
    HeterogeneousMatrix Filter(5, 5);

    HeterogeneousMatrix C(M, N);

    float filter[25] = {
        0.01f, 0.02f, 0.04f, 0.02f, 0.01f,
        0.02f, 0.05f, 0.08f, 0.05f, 0.02f,
        0.04f, 0.08f, 0.12f, 0.08f, 0.04f,
        0.02f, 0.05f, 0.08f, 0.05f, 0.02f,
        0.01f, 0.02f, 0.04f, 0.02f, 0.01f
    };

    for (int i = 0; i < 25; i++)
        Filter.h_data[i] = filter[i];

    // Generate input matrices:
    initialize_matrix_hw3(A);

    A.to_device();
    Filter.to_device();

    // Timing done based on the following tutorial https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolution_v1<<< grid_size, block_size >>>(A.d_data, Filter.d_data ,C.d_data, M, N, 2);
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
        //
        int block_x = block_size.x, block_y = block_size.y;
        dim3 grid_size(
          (N + block_x - 1) / block_x,
          (M + block_y - 1) / block_y
        );

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }

}
