#ifndef HW2_UTILS_H
#define HW2_UTILS_H

#include <iostream>
#include <string>


void cuda_check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}


std::string dim3_to_string(dim3 value) {
    return std::string("(")
        + std::to_string(value.x) + ", "
        + std::to_string(value.y) + ", "
        + std::to_string(value.z) + ")";
}

struct HeterogeneousMatrix {
    unsigned int n_rows;
    unsigned int n_cols;
    float* h_data;
    float* d_data;

    HeterogeneousMatrix(const unsigned int n_rows, const unsigned int n_cols)
        : n_rows(n_rows), n_cols(n_cols), h_data(new float[n_rows * n_cols]) {
        cuda_check(cudaMalloc( (void**) &d_data, n_rows * n_cols * sizeof(float)));
    }

    void to_device() {
        cuda_check(
            cudaMemcpy(d_data, h_data, n_rows * n_cols * sizeof(float), cudaMemcpyHostToDevice)
            );
    }

    void to_host() {
        cuda_check(
            cudaMemcpy(h_data, d_data, n_rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost)
            );
    }

    ~HeterogeneousMatrix() {
        delete[] h_data;
        cuda_check(
            cudaFree(d_data)
            );
    }

    // todo: write other constructors to follow the rule of 5

};


void initialize_matrix_hw3(HeterogeneousMatrix& matrix) {
    auto n_rows = matrix.n_rows;
    auto n_cols = matrix.n_cols;
    auto data = matrix.h_data;

    // go over each row:
    for (int r = 0; r < n_rows; r++) {
        // go over each column
        for (int c = 0; c < n_cols; c++)
            // wanna do: A[r][c] = ... A[r][c] is equivalent to A[r * width + c]
            data[r * n_cols + c] = (r % 2 == 0) ? 0.5f : 0.4f;
    }
}

void print_matrix_hw3(HeterogeneousMatrix& matrix) {

    auto n_rows = matrix.n_rows;
    auto n_cols = matrix.n_cols;
    auto data = matrix.h_data;

    for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++)
            std::cout << data[r * n_cols + c] << "  ";
        std::cout << std::endl << std::endl;
    }

}

#endif //HW2_UTILS_H
