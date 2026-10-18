#include "utils.h"

__constant__  float F[25];

template<int R, int IN_TILE_SIZE, int OUT_TILE_SIZE>
__global__ void convolution_v1(
    float* A, // [M,N]
    float* B, // [M,N]
    int M,
    int N
) {

    //load phase
    __shared__ float sA[IN_TILE_SIZE][IN_TILE_SIZE];

    int load_row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y - R,
        load_col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x - R;

    sA[threadIdx.y][threadIdx.x] =
        ( load_row > -1 && load_row < M && load_col > -1 && load_col < N )
        ? A[ load_row * N + load_col ] : 0.0f;

    __syncthreads();

    // Now the compute phase:
    float sum = 0.0f;
    if (threadIdx.x < OUT_TILE_SIZE && threadIdx.y < OUT_TILE_SIZE) {

        for (int i = 0; i < (2*R + 1); i++ ) {
            for (int j = 0; j < (2*R + 1); j++)
                sum += sA[threadIdx.y + i][threadIdx.x + j] * F[i * (2*R + 1) + j];
        }

        int out_row = blockIdx.y * OUT_TILE_SIZE + threadIdx.y,
            out_col = blockIdx.x * OUT_TILE_SIZE + threadIdx.x;

        if (out_row < M && out_col < N)
            B[out_row * N + out_col] = sum;
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

        case 32:
            convolution_v1<2, 32, 32 - 2*2><<<grid_size, block_size>>>(A.d_data, C.d_data, M, N);
            break;
        case 16:
            convolution_v1<2, 16, 12><<<grid_size, block_size>>>(A.d_data, C.d_data, M, N);
            break;
        case 8:
            convolution_v1<2, 8, 4><<<grid_size, block_size>>>(A.d_data, C.d_data, M, N);
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
        dim3(32, 32, 1),
    };

    int R = 2;
    for (auto block_size : block_sizes) {
        //
        int block_x = block_size.x, block_y = block_size.y;
        int out_tile_x = block_x - 2 * R, out_tile_y = block_y - 2 * R;
        dim3 grid_size(
            (N + out_tile_x - 1) / out_tile_x,
            (M + out_tile_y - 1) / out_tile_y
        );

        std::cout << "block size: " << dim3_to_string(block_size) << " , grid_size" << dim3_to_string(grid_size) << std::endl;

        float elapsed_time = block_size_experiments(M, N, grid_size, block_size);
        std::cout << "kernel elapsed time: " << elapsed_time << "ms" << std::endl;

        std::cout << std::endl << "--------" << std::endl;

    }

}
