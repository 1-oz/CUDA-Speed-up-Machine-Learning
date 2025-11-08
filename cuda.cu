#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <zlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define IMAGE_SIZE 28 * 28
#define NUM_CLASSES 10

#define TILE_SIZE 16


float* load_mnist_images(const char* filename, int* num_images) {
    gzFile file = gzopen(filename, "rb");

    unsigned char header[16];
    gzread(file, header, 16);

    int count = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];
    *num_images = count;

    float* data = (float*)malloc(count * IMAGE_SIZE * sizeof(float));

    unsigned char* buffer = (unsigned char*)malloc(count * IMAGE_SIZE);
    gzread(file, buffer, count * IMAGE_SIZE);
    for (int i = 0; i < count * IMAGE_SIZE; i++) {
        data[i] = buffer[i] / 255.0f;
    }

    free(buffer);
    gzclose(file);
    return data;
}

float* load_mnist_labels(const char* filename, int* num_labels) {
    gzFile file = gzopen(filename, "rb");

    unsigned char header[8];
    gzread(file, header, 8);

    int count = (header[4] << 24) | (header[5] << 16) | (header[6] << 8) | header[7];
    *num_labels = count;

    unsigned char* buffer = (unsigned char*)malloc(count);
    gzread(file, buffer, count);

    float* labels = (float*)calloc(count * NUM_CLASSES, sizeof(float));

    for (int i = 0; i < count; i++) {
        labels[i * NUM_CLASSES + buffer[i]] = 1.0f;
    }

    free(buffer);
    gzclose(file);
    return labels;
}


// ReLU activation Func
void relu(float* output, float* input, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

__global__ void relu_kernel(float* Z, float* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmaxf(0.0f, Z[idx]);
    }
}

//Softmax
void softmax(float* x, float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(x[i * cols + j]);
            sum_exp += output[i * cols + j];
        }
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum_exp;
        }
    }
}

__global__ void softmax_kernel(float* Z, float* A, int batch_size, int output_size) {
    int batch_idx = blockIdx.x; 
    int idx = threadIdx.x; 

    if (idx < output_size) {
        float max_val = -1e10;
        for (int i = 0; i < output_size; i++) {
            max_val = fmaxf(max_val, Z[batch_idx * output_size + i]);
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < output_size; i++) {
            sum_exp += expf(Z[batch_idx * output_size + i] - max_val);
        }

        A[batch_idx * output_size + idx] = expf(Z[batch_idx * output_size + idx] - max_val) / sum_exp;
    }
}


//normal distr rand Func
//Reference: https://github.com/ingenthr/memcachetest/blob/master/boxmuller.c
float randn() {
    static int has_spare = 0;
    static float spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    has_spare = 1;
    float u, v, s;
    do {
        u = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        v = (rand() / ((float)RAND_MAX)) * 2.0f - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    s = sqrtf(-2.0f * logf(s) / s);
    spare = v * s;
    return u * s;
}

// Init W's Kaiming He
void initialize_weights(int input_size, int hidden1, int hidden2, int output_size, 
                        float** W1, float** b1, float** W2, float** b2, float** W3, float** b3) {

    *W1 = (float*)malloc(input_size * hidden1 * sizeof(float));
    *b1 = (float*)calloc(hidden1, sizeof(float));

    *W2 = (float*)malloc(hidden1 * hidden2 * sizeof(float));
    *b2 = (float*)calloc(hidden2, sizeof(float));

    *W3 = (float*)malloc(hidden2 * output_size * sizeof(float));
    *b3 = (float*)calloc(output_size, sizeof(float));

    float std_W1 = sqrtf(6.0f / input_size);
    float std_W2 = sqrtf(6.0f / hidden1);
    float std_W3 = sqrtf(6.0f / hidden2);

    for (int i = 0; i < input_size * hidden1; i++) (*W1)[i] = std_W1 * randn();
    for (int i = 0; i < hidden1 * hidden2; i++) (*W2)[i] = std_W2 * randn();
    for (int i = 0; i < hidden2 * output_size; i++) (*W3)[i] = std_W3 * randn();
}

__global__ void gemm_cuda_naive(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        if (bias != NULL) {
            C[row * N + col] += bias[col];
        }
    }
}

__global__ void gemm_cuda_tiled(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int tile_idx = 0; tile_idx < (K + TILE_SIZE - 1) / TILE_SIZE; tile_idx++) {
        if (row < M && (tile_idx * TILE_SIZE + threadIdx.x) < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + tile_idx * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (tile_idx * TILE_SIZE + threadIdx.y) < K) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile_idx * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum + (bias ? bias[col] : 0.0f);
    }
}

void gemm_cuda(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    float *d_A, *d_B, *d_C, *d_bias;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    if (bias != NULL) {
        cudaMalloc((void**)&d_bias, N * sizeof(float));
        cudaMemcpy(d_bias, bias, N * sizeof(float), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_cuda_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, bias ? d_bias : NULL, M, N, K);

    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    if (bias != NULL) cudaFree(d_bias);
}

void gemm_cuda_device(float* d_A, float* d_B, float* d_C, float* d_bias, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_cuda_tiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, d_bias, M, N, K);
}

void forward_propagation(float* X, float* W1, float* b1, float* W2, float* b2, 
                         float* W3, float* b3, int batch_size, int input_size, 
                         int hidden1, int hidden2, int output_size,
                         float* Z1, float* A1, float* Z2, float* A2, float* Z3, float* A3) {
    
    // Z1 = X * W1 + b1
    gemm_cuda(X, W1, Z1, b1, batch_size, hidden1, input_size);
    relu(A1, Z1, batch_size * hidden1);

    // Z2 = A1 * W2 + b2
    gemm_cuda(A1, W2, Z2, b2, batch_size, hidden2, hidden1);
    relu(A2, Z2, batch_size * hidden2);

    // Z3 = A2 * W3 + b3
    gemm_cuda(A2, W3, Z3, b3, batch_size, output_size, hidden2);

    // A3 = Softmax(Z3)
    softmax(Z3, A3, batch_size, output_size);
}

void forward_propagation_gpu(float* d_X, float* d_W1, float* d_b1, float* d_W2, float* d_b2, 
                             float* d_W3, float* d_b3, int batch_size, int input_size, 
                             int hidden1, int hidden2, int output_size,
                             float* d_Z1, float* d_A1, float* d_Z2, float* d_A2, float* d_Z3, float* d_A3) {
    
    dim3 blockDim(256);
    // Z1 = X * W1 + b1
    gemm_cuda_device(d_X, d_W1, d_Z1, d_b1, batch_size, hidden1, input_size);
    relu_kernel<<<(batch_size * hidden1 + 255) / 256, blockDim>>>(d_Z1, d_A1, batch_size * hidden1);

    // Z2 = A1 * W2 + b2
    gemm_cuda_device(d_A1, d_W2, d_Z2, d_b2, batch_size, hidden2, hidden1);
    relu_kernel<<<(batch_size * hidden2 + 255) / 256, blockDim>>>(d_Z2, d_A2, batch_size * hidden2);

    // Z3 = A2 * W3 + b3
    gemm_cuda_device(d_A2, d_W3, d_Z3, d_b3, batch_size, output_size, hidden2);
    softmax_kernel<<<batch_size, output_size>>>(d_Z3, d_A3, batch_size, output_size);
}

__global__ void add_bias_kernel(float* Z, float* b, int batch_size, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size * hidden_dim) {
        int col = idx % hidden_dim;
        Z[idx] += b[col];
    }
}

void forward_propagation_b(cublasHandle_t handle, float* d_X, float* d_W1, float* d_b1, float* d_W2, float* d_b2, 
                             float* d_W3, float* d_b3, int batch_size, int input_size, 
                             int hidden1, int hidden2, int output_size,
                             float* d_Z1, float* d_A1, float* d_Z2, float* d_A2, float* d_Z3, float* d_A3) {

    float alpha = 1.0f;
    float beta = 0.0f;

    // Z1 = X * W1 + b1
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden1, batch_size, input_size,
                &alpha, d_W1, hidden1, d_X, input_size,
                &beta, d_Z1, hidden1);
    add_bias_kernel<<<(batch_size * hidden1 + 255) / 256, 256>>>(d_Z1, d_b1, batch_size, hidden1);
    relu_kernel<<<(batch_size * hidden1 + 255) / 256, 256>>>(d_Z1, d_A1, batch_size * hidden1);

    // Z2 = A1 * W2 + b2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                hidden2, batch_size, hidden1,
                &alpha, d_W2, hidden2, d_A1, hidden1,
                &beta, d_Z2, hidden2);
    add_bias_kernel<<<(batch_size * hidden2 + 255) / 256, 256>>>(d_Z2, d_b2, batch_size, hidden2);
    relu_kernel<<<(batch_size * hidden2 + 255) / 256, 256>>>(d_Z2, d_A2, batch_size * hidden2);

    // Z3 = A2 * W3 + b3
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                output_size, batch_size, hidden2,
                &alpha, d_W3, output_size, d_A2, hidden2,
                &beta, d_Z3, output_size);
    add_bias_kernel<<<(batch_size * output_size + 255) / 256, 256>>>(d_Z3, d_b3, batch_size, output_size);
    softmax_kernel<<<batch_size, output_size>>>(d_Z3, d_A3, batch_size, output_size);
}


// ReLU Derivative Func
void relu_derivative(float* x, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (x[i] > 0) ? 1.0f : 0.0f;
    }
}
//Mat transpose
void transpose(float* A, float* B, int m, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}


void elementwise_multiply(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
}

void column_sum(float* A, float* result, int m, int n) {
    memset(result, 0, n * sizeof(float));

    for (int j = 0; j < n; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[i * n + j];
        }

        result[j] = sum / m;
    }
}

__global__ void relu_derivative_kernel(float* x, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (x[idx] > 0) ? 1.0f : 0.0f;
    }
}

__global__ void transpose_kernel(float* A, float* B, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        B[col * m + row] = A[row * n + col];
    }
}

__global__ void elementwise_multiply_kernel(float* A, float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void column_sum_kernel(float* A, float* result, int m, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) {
            sum += A[i * n + col];
        }
        result[col] = sum / m;
    }
}

//backward propa specific kernels
__global__ void compute_dZ3_kernel(float* A3, float* Y, float* dZ3, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dZ3[idx] = (A3[idx] - Y[idx]);
    }
}

__global__ void compute_dZ2_kernel(float* dZ3, float* W3_T, float* dZ2, float* Z2, int batch_size, int hidden2, int output_size) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    if (batch_idx < batch_size && hidden_idx < hidden2) {
        float sum = 0.0f;
        for (int j = 0; j < output_size; j++) {
            sum += dZ3[batch_idx * output_size + j] * W3_T[hidden_idx * output_size + j];
        }

        int index = batch_idx * hidden2 + hidden_idx;
        dZ2[index] = (Z2[index] > 0) ? sum : 0.0f;
    }
}


__global__ void compute_dZ1_kernel(float* dZ2, float* W2_T, float* dZ1, float* Z1, int batch_size, int hidden1, int hidden2) {
    int batch_idx = blockIdx.x;
    int hidden_idx = threadIdx.x;

    if (batch_idx < batch_size && hidden_idx < hidden1) {
        float sum = 0.0f;
        for (int j = 0; j < hidden2; j++) {
            sum += dZ2[batch_idx * hidden2 + j] * W2_T[hidden_idx * hidden2 + j];
        }

        int index = batch_idx * hidden1 + hidden_idx;
        dZ1[index] = (Z1[index] > 0) ? sum : 0.0f;
    }
}

__global__ void update_parameters_kernel(float* W, float* dW, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        W[idx] -= lr * dW[idx];  // W = W - lr * dW
    }
}

__global__ void update_bias_kernel(float* b, float* db, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] -= lr * db[idx];  // b = b - lr * db
    }
}

__global__ void scale_kernel(float* d_matrix, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_matrix[idx] /= batch_size;
    }
}



void backward_propagation(float* X, float* Y, float* Z1, float* A1, float* Z2, float* A2, 
                          float* Z3, float* A3, float* W1, float* W2, float* W3,
                          int batch_size, int input_size, int hidden1, int hidden2, int output_size,
                          float* dW1, float* db1, float* dW2, float* db2, float* dW3, float* db3) {

    // dZ3 = A3 - Y
    float* dZ3 = (float*)malloc(batch_size * output_size * sizeof(float));
    for (int i = 0; i < batch_size * output_size; i++) {
        dZ3[i] = A3[i] - Y[i];
    }

    // dW3 = (A2^T * dZ3) / batch_size
    float* A2_T = (float*)malloc(hidden2 * batch_size * sizeof(float));
    transpose(A2, A2_T, batch_size, hidden2);
    gemm_cuda(A2_T, dZ3, dW3, NULL, hidden2, output_size, batch_size); 
    for (int i = 0; i < hidden2 * output_size; i++) dW3[i] /= batch_size;
    free(A2_T);

    // db3 = sum(dZ3, axis=0) / batch_size
    column_sum(dZ3, db3, batch_size, output_size);

    // dZ2 = (dZ3 * W3^T) ⊙ ReLU'(Z2)
    float* W3_T = (float*)malloc(output_size * hidden2 * sizeof(float));
    transpose(W3, W3_T, hidden2, output_size);
    float* dZ2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    gemm_cuda(dZ3, W3_T, dZ2, NULL, batch_size, hidden2, output_size);
    float* dReLU_Z2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    relu_derivative(Z2, dReLU_Z2, batch_size * hidden2);
    elementwise_multiply(dZ2, dReLU_Z2, dZ2, batch_size * hidden2);
    free(W3_T);
    free(dZ3);
    free(dReLU_Z2);

    // dW2 = (A1^T * dZ2) / batch_size
    float* A1_T = (float*)malloc(hidden1 * batch_size * sizeof(float));
    transpose(A1, A1_T, batch_size, hidden1);
    gemm_cuda(A1_T, dZ2, dW2, NULL, hidden1, hidden2, batch_size);
    for (int i = 0; i < hidden1 * hidden2; i++) dW2[i] /= batch_size;
    free(A1_T);

    // db2 = sum(dZ2, axis=0) / batch_size
    column_sum(dZ2, db2, batch_size, hidden2);

    // dZ1 = (dZ2 * W2^T) ⊙ ReLU'(Z1)
    float* W2_T = (float*)malloc(hidden2 * hidden1 * sizeof(float));
    transpose(W2, W2_T, hidden1, hidden2);
    float* dZ1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    gemm_cuda(dZ2, W2_T, dZ1, NULL, batch_size, hidden1, hidden2);
    float* dReLU_Z1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    relu_derivative(Z1, dReLU_Z1, batch_size * hidden1);
    elementwise_multiply(dZ1, dReLU_Z1, dZ1, batch_size * hidden1);
    free(W2_T);
    free(dZ2);
    free(dReLU_Z1);

    // dW1 = (X^T * dZ1) / batch_size
    float* X_T = (float*)malloc(input_size * batch_size * sizeof(float));
    transpose(X, X_T, batch_size, input_size);
    gemm_cuda(X_T, dZ1, dW1, NULL, input_size, hidden1, batch_size);
    for (int i = 0; i < input_size * hidden1; i++) dW1[i] /= batch_size;
    free(X_T);

    // db1 = sum(dZ1, axis=0) / batch_size
    column_sum(dZ1, db1, batch_size, hidden1);

    free(dZ1);
}

void backward_propagation_gpu(float* d_X, float* d_Y, float* d_Z1, float* d_A1, float* d_Z2, float* d_A2, 
                              float* d_Z3, float* d_A3, float* d_W1, float* d_W2, float* d_W3,float* d_b1, float* d_b2, float* d_b3,
                              int batch_size, int input_size, int hidden1, int hidden2, int output_size,
                              float* d_dW1, float* d_db1, float* d_dW2, float* d_db2, float* d_dW3, float* d_db3,
                              float* d_dZ1, float* d_dZ2, float* d_dZ3, float* d_A2_T, float* d_W3_T, float* d_A1_T, float* d_W2_T, float* d_X_T,
                              float learning_rate) {

    dim3 blockDim(256);

    // dZ3 = A3 - Y
    dim3 gridDim_dZ3((batch_size * output_size + 255) / 256);
    compute_dZ3_kernel<<<gridDim_dZ3, blockDim>>>(d_A3, d_Y, d_dZ3, batch_size * output_size);
    
    
    //  dW3 = (A2^T * dZ3) / batch_size
    dim3 gridDim_transpose((hidden2 + 15) / 16, (batch_size + 15) / 16);
    dim3 blockDim_transpose(16, 16);
    transpose_kernel<<<gridDim_transpose, blockDim_transpose>>>(d_A2, d_A2_T, batch_size, hidden2);

    dim3 gridDim_dW3((hidden2 * output_size + 255) / 256);
    gemm_cuda_device(d_A2_T, d_dZ3, d_dW3, NULL, hidden2, output_size, batch_size);
    scale_kernel<<<gridDim_dW3, blockDim>>>(d_dW3, batch_size, hidden2 * output_size);
    
    //db3 = sum(dZ3, axis=0) / batch_size
    dim3 gridDim_db3((output_size + 255) / 256);
    column_sum_kernel<<<gridDim_db3, blockDim>>>(d_dZ3, d_db3, batch_size, output_size);
    scale_kernel<<<gridDim_db3, blockDim>>>(d_db3, batch_size, output_size);

    // dZ2 = (dZ3 * W3^T) ⊙ ReLU'(Z2)
    // W3^T
    dim3 gridDim_transpose_W3((hidden2 + 15) / 16, (output_size + 15) / 16);
    dim3 blockDim_transpose_W3(16, 16);
    transpose_kernel<<<gridDim_transpose_W3, blockDim_transpose_W3>>>(d_W3, d_W3_T, hidden2, output_size);
    //dZ2 = (dZ3 * W3^T) ⊙ ReLU'(Z2)
    dim3 gridDim_dZ2(batch_size);
    dim3 blockDim_dZ2(hidden2);
    compute_dZ2_kernel<<<gridDim_dZ2, blockDim_dZ2>>>(d_dZ3, d_W3_T, d_dZ2, d_A2, batch_size, hidden2, output_size);

    // 5.dW2 = (A1_T * dZ2) / batch_size
    dim3 gridDim_transpose_A1((hidden1 + 15) / 16, (batch_size + 15) / 16);
    transpose_kernel<<<gridDim_transpose_A1, blockDim_transpose>>>(d_A1, d_A1_T, batch_size, hidden1);

    dim3 gridDim_dW2((hidden1 * hidden2 + 255) / 256);
    gemm_cuda_device(d_A1_T, d_dZ2, d_dW2, NULL, hidden1, hidden2, batch_size);
    scale_kernel<<<gridDim_dW2, blockDim>>>(d_dW2, batch_size, hidden1 * hidden2);
    

    // db2 = sum(dZ2, axis=0) / batch_size
    dim3 gridDim_db2((hidden2 + 255) / 256);
    column_sum_kernel<<<gridDim_db2, blockDim>>>(d_dZ2, d_db2, batch_size, hidden2);
    scale_kernel<<<gridDim_db2, blockDim>>>(d_db2, batch_size, hidden2);

    //dZ1 = (dZ2 * W2^T) ⊙ ReLU'(Z1)
    //W2^T
    dim3 gridDim_transpose_W2((hidden1 + 15) / 16, (hidden2 + 15) / 16);
    dim3 blockDim_transpose_W2(16, 16);
    transpose_kernel<<<gridDim_transpose_W2, blockDim_transpose_W2>>>(d_W2, d_W2_T, hidden1, hidden2);
    //dZ1 = (dZ2 * W2^T) ⊙ ReLU'(Z1)
    dim3 gridDim_dZ1(batch_size);
    dim3 blockDim_dZ1(hidden1);
    compute_dZ1_kernel<<<gridDim_dZ1, blockDim_dZ1>>>(d_dZ2, d_W2_T, d_dZ1, d_A1, batch_size, hidden1, hidden2);

    // dW1 = (X_T * dZ1) / batch_size
    transpose_kernel<<<gridDim_transpose, blockDim_transpose>>>(d_X, d_X_T, batch_size, input_size);

    dim3 gridDim_dW1((input_size * hidden1 + 255) / 256);
    gemm_cuda_device(d_X_T, d_dZ1, d_dW1, NULL, input_size, hidden1, batch_size);
    scale_kernel<<<gridDim_dW1, blockDim>>>(d_dW1, batch_size, input_size * hidden1);

    dim3 gridDim_db1((hidden1 + 255) / 256);
    column_sum_kernel<<<gridDim_db1, blockDim>>>(d_dZ1, d_db1, batch_size, hidden1);
    scale_kernel<<<gridDim_db1, blockDim>>>(d_db1, batch_size, hidden1);

    update_parameters_kernel<<<gridDim_dW1, blockDim>>>(d_W1, d_dW1, learning_rate, input_size * hidden1);
    update_parameters_kernel<<<gridDim_dW2, blockDim>>>(d_W2, d_dW2, learning_rate, hidden1 * hidden2);
    update_parameters_kernel<<<gridDim_dW3, blockDim>>>(d_W3, d_dW3, learning_rate, hidden2 * output_size);

    update_bias_kernel<<<gridDim_db1, blockDim>>>(d_b1, d_db1, learning_rate, hidden1);
    update_bias_kernel<<<gridDim_db2, blockDim>>>(d_b2, d_db2, learning_rate, hidden2);
    update_bias_kernel<<<gridDim_db3, blockDim>>>(d_b3, d_db3, learning_rate, output_size);

    cudaDeviceSynchronize();
}

void backward_propagation_b(cublasHandle_t handle, 
                              float* d_X, float* d_Y, float* d_Z1, float* d_A1, float* d_Z2, float* d_A2, 
                              float* d_Z3, float* d_A3, float* d_W1, float* d_W2, float* d_W3, float* d_b1, float* d_b2, float* d_b3,
                              int batch_size, int input_size, int hidden1, int hidden2, int output_size,
                              float* d_dW1, float* d_db1, float* d_dW2, float* d_db2, float* d_dW3, float* d_db3,
                              float* d_dZ1, float* d_dZ2, float* d_dZ3, float* d_A2_T, float* d_W3_T, float* d_A1_T, float* d_W2_T, float* d_X_T,
                              float learning_rate) {
    
    float alpha = 1.0f, beta = 0.0f;

    dim3 gridDim_dZ3((batch_size * output_size + 255) / 256);
    compute_dZ3_kernel<<<gridDim_dZ3, 256>>>(d_A3, d_Y, d_dZ3, batch_size * output_size);

    dim3 gridDim_transpose((hidden2 + 15) / 16, (batch_size + 15) / 16);
    transpose_kernel<<<gridDim_transpose, dim3(16, 16)>>>(d_A2, d_A2_T, batch_size, hidden2);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden2, output_size, batch_size, &alpha, d_A2_T, hidden2, d_dZ3, batch_size, &beta, d_dW3, hidden2);
    scale_kernel<<<(hidden2 * output_size + 255) / 256, 256>>>(d_dW3, batch_size, hidden2 * output_size);

    column_sum_kernel<<<(output_size + 255) / 256, 256>>>(d_dZ3, d_db3, batch_size, output_size);
    scale_kernel<<<(output_size + 255) / 256, 256>>>(d_db3, batch_size, output_size);

    transpose_kernel<<<gridDim_transpose, dim3(16, 16)>>>(d_W3, d_W3_T, output_size, hidden2);
    dim3 gridDim_dZ2(batch_size);
    dim3 blockDim_dZ2(hidden2);
    compute_dZ2_kernel<<<gridDim_dZ2, blockDim_dZ2>>>(d_dZ3, d_W3_T, d_dZ2, d_A2, batch_size, hidden2, output_size);

    transpose_kernel<<<gridDim_transpose, dim3(16, 16)>>>(d_A1, d_A1_T, batch_size, hidden1);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, hidden1, hidden2, batch_size, &alpha, d_A1_T, hidden1, d_dZ2, batch_size, &beta, d_dW2, hidden1);
    scale_kernel<<<(hidden1 * hidden2 + 255) / 256, 256>>>(d_dW2, batch_size, hidden1 * hidden2);

    column_sum_kernel<<<(hidden2 + 255) / 256, 256>>>(d_dZ2, d_db2, batch_size, hidden2);
    scale_kernel<<<(hidden2 + 255) / 256, 256>>>(d_db2, batch_size, hidden2);

    transpose_kernel<<<gridDim_transpose, dim3(16, 16)>>>(d_W2, d_W2_T, hidden2, hidden1);
    dim3 gridDim_dZ1(batch_size);
    dim3 blockDim_dZ1(hidden1);
    compute_dZ1_kernel<<<gridDim_dZ1, blockDim_dZ1>>>(d_dZ2, d_W2_T, d_dZ1, d_A1, batch_size, hidden1, hidden2);

    transpose_kernel<<<gridDim_transpose, dim3(16, 16)>>>(d_X, d_X_T, batch_size, input_size);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, input_size, hidden1, batch_size, &alpha, d_X_T, input_size, d_dZ1, batch_size, &beta, d_dW1, input_size);
    scale_kernel<<<(input_size * hidden1 + 255) / 256, 256>>>(d_dW1, batch_size, input_size * hidden1);

    column_sum_kernel<<<(hidden1 + 255) / 256, 256>>>(d_dZ1, d_db1, batch_size, hidden1);
    scale_kernel<<<(hidden1 + 255) / 256, 256>>>(d_db1, batch_size, hidden1);

    update_parameters_kernel<<<(input_size * hidden1 + 255) / 256, 256>>>(d_W1, d_dW1, learning_rate, input_size * hidden1);
    update_parameters_kernel<<<(hidden1 * hidden2 + 255) / 256, 256>>>(d_W2, d_dW2, learning_rate, hidden1 * hidden2);
    update_parameters_kernel<<<(hidden2 * output_size + 255) / 256, 256>>>(d_W3, d_dW3, learning_rate, hidden2 * output_size);

    update_bias_kernel<<<(hidden1 + 255) / 256, 256>>>(d_b1, d_db1, learning_rate, hidden1);
    update_bias_kernel<<<(hidden2 + 255) / 256, 256>>>(d_b2, d_db2, learning_rate, hidden2);
    update_bias_kernel<<<(output_size + 255) / 256, 256>>>(d_b3, d_db3, learning_rate, output_size);

    cudaDeviceSynchronize();
}


void update_parameters(float* W1, float* b1, float* W2, float* b2, float* W3, float* b3,
                       float* dW1, float* db1, float* dW2, float* db2, float* dW3, float* db3,
                       int input_size, int hidden1, int hidden2, int output_size, float learning_rate) {
    
    // W1
    for (int i = 0; i < input_size * hidden1; i++) {
        W1[i] -= learning_rate * dW1[i];
    }
    for (int i = 0; i < hidden1; i++) {
        b1[i] -= learning_rate * db1[i];
    }

    // W2
    for (int i = 0; i < hidden1 * hidden2; i++) {
        W2[i] -= learning_rate * dW2[i];
    }
    for (int i = 0; i < hidden2; i++) {
        b2[i] -= learning_rate * db2[i];
    }

    // W3
    for (int i = 0; i < hidden2 * output_size; i++) {
        W3[i] -= learning_rate * dW3[i];
    }
    for (int i = 0; i < output_size; i++) {
        b3[i] -= learning_rate * db3[i];
    }
}


void swap(float* a, float* b, int size) {
    float temp;
    for (int i = 0; i < size; i++) {
        temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

//Shuffle Func for SGD
void shuffle_data(float* X, float* Y, int samples, int input_size, int output_size) {
    int* indices = (int*)malloc(samples * sizeof(int));
    for (int i = 0; i < samples; i++) {
        indices[i] = i;
    }

    for (int i = samples - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        swap(&X[i * input_size], &X[j * input_size], input_size);
        swap(&Y[i * output_size], &Y[j * output_size], output_size);
    }
    free(indices);
}

//Metrics
float compute_accuracy(float* A3, float* Y_test, int samples, int output_size) {
    int correct = 0;
    for (int i = 0; i < samples; i++) {
        int predicted_label = 0;
        int actual_label = 0;
        float max_pred = A3[i * output_size];

        for (int j = 1; j < output_size; j++) {
            if (A3[i * output_size + j] > max_pred) {
                max_pred = A3[i * output_size + j];
                predicted_label = j;
            }
            if (Y_test[i * output_size + j] == 1.0f) {
                actual_label = j;
            }
        }

        if (predicted_label == actual_label) {
            correct++;
        }
    }
    return (float)correct / samples;
}

float compute_loss(float* A3, float* Y, int samples, int output_size) {
    float loss = 0.0f;
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < output_size; j++) {
            if (Y[i * output_size + j] > 0) {
                loss += -logf(A3[i * output_size + j] + 1e-9);
            }
        }
    }
    return loss / samples;
}

void save_loss_to_binary(float* loss, int epochs, const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(loss, sizeof(float), epochs, file);
    fclose(file);
}

void train_b(float* X_train, float* Y_train, float* X_val, float* Y_val, float* X_test, float* Y_test, 
            int train_samples, int val_samples, int test_samples, 
            int input_size, int hidden1, int hidden2, int output_size, int epochs, int batch_size, float learning_rate) {

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float *W1, *b1, *W2, *b2, *W3, *b3;
    initialize_weights(input_size, hidden1, hidden2, output_size, &W1, &b1, &W2, &b2, &W3, &b3);

    float* loss_history = (float*)malloc(epochs * sizeof(float));

    float *d_X_train, *d_Y_train, *d_X, *d_X_val, *d_Y, *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    float *d_Z1, *d_A1, *d_Z2, *d_A2, *d_Z3, *d_A3;
    float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dW3, *d_db3;
    
    cudaMalloc(&d_X, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_X_val, val_samples * input_size * sizeof(float));
    cudaMalloc(&d_Y, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_W1, input_size * hidden1 * sizeof(float));
    cudaMalloc(&d_b1, hidden1 * sizeof(float));
    cudaMalloc(&d_W2, hidden1 * hidden2 * sizeof(float));
    cudaMalloc(&d_b2, hidden2 * sizeof(float));
    cudaMalloc(&d_W3, hidden2 * output_size * sizeof(float));
    cudaMalloc(&d_b3, output_size * sizeof(float));
    cudaMalloc(&d_Z1, batch_size * hidden1 * sizeof(float));
    cudaMalloc(&d_A1, batch_size * hidden1 * sizeof(float));
    cudaMalloc(&d_Z2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_A2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_Z3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_A3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_dW1, input_size * hidden1 * sizeof(float));
    cudaMalloc(&d_db1, hidden1 * sizeof(float));
    cudaMalloc(&d_dW2, hidden1 * hidden2 * sizeof(float));
    cudaMalloc(&d_db2, hidden2 * sizeof(float));
    cudaMalloc(&d_dW3, hidden2 * output_size * sizeof(float));
    cudaMalloc(&d_db3, output_size * sizeof(float));

    cudaMemcpy(d_X_val, X_val, val_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, input_size * hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hidden1 * hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, hidden2 * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3, output_size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_dZ3, *d_dZ2, *d_dZ1;
    cudaMalloc(&d_dZ3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_dZ2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_dZ1, batch_size * hidden1 * sizeof(float));
    float* d_A2_T, * d_W3_T, * d_A1_T,* d_W2_T, * d_X_T;
    cudaMalloc(&d_A2_T, hidden2 * batch_size * sizeof(float));
    cudaMalloc(&d_W3_T, output_size * hidden2 * sizeof(float));
    cudaMalloc(&d_A1_T, hidden1 * batch_size * sizeof(float));
    cudaMalloc(&d_W2_T, hidden2 * hidden1 * sizeof(float));
    cudaMalloc(&d_X_T, input_size * batch_size * sizeof(float));

    float *d_Z1_val, *d_A1_val, *d_Z2_val, *d_A2_val, *d_Z3_val, *d_A3_val;
    cudaMalloc(&d_Z1_val, val_samples * hidden1 * sizeof(float));
    cudaMalloc(&d_A1_val, val_samples * hidden1 * sizeof(float));
    cudaMalloc(&d_Z2_val, val_samples * hidden2 * sizeof(float));
    cudaMalloc(&d_A2_val, val_samples * hidden2 * sizeof(float));
    cudaMalloc(&d_Z3_val, val_samples * output_size * sizeof(float));
    cudaMalloc(&d_A3_val, val_samples * output_size * sizeof(float));

    cudaMalloc(&d_X_train, train_samples * input_size * sizeof(float));
    cudaMalloc(&d_Y_train, train_samples * output_size * sizeof(float));
    cudaMemcpy(d_X_train, X_train, train_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_train, Y_train, train_samples * output_size * sizeof(float), cudaMemcpyHostToDevice);


    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_data(X_train, Y_train, train_samples, input_size, output_size);

        for (int i = 0; i < train_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > train_samples) ? (train_samples - i) : batch_size;

            float* d_X = d_X_train + i * input_size;
            float* d_Y = d_Y_train + i * output_size;

            forward_propagation_b(handle,d_X, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                                    current_batch_size, input_size, hidden1, hidden2, output_size,
                                    d_Z1, d_A1, d_Z2, d_A2, d_Z3, d_A3);

            backward_propagation_gpu(d_X, d_Y, d_Z1, d_A1, d_Z2, d_A2, d_Z3, d_A3,
                                     d_W1, d_W2, d_W3, d_b1, d_b2, d_b3,
                                     current_batch_size, input_size, hidden1, hidden2, output_size,
                                     d_dW1, d_db1, d_dW2, d_db2, d_dW3, d_db3, d_dZ1, d_dZ2, d_dZ3,
                                     d_A2_T, d_W3_T, d_A1_T,d_W2_T, d_X_T,
                                     learning_rate);

            cudaDeviceSynchronize();
        }

        forward_propagation_b(handle, d_X_val, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                                val_samples, input_size, hidden1, hidden2, output_size,
                                d_Z1_val, d_A1_val, d_Z2_val, d_A2_val, d_Z3_val, d_A3_val);

        float* A3_val = (float*)malloc(val_samples * output_size * sizeof(float));
        cudaMemcpy(A3_val, d_A3_val, val_samples * output_size * sizeof(float), cudaMemcpyDeviceToHost);
        loss_history[epoch] = compute_loss(A3_val, Y_val, val_samples, output_size);
        printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, loss_history[epoch]);

        free(A3_val);
    }
    cudaFree(d_Z1_val); cudaFree(d_A1_val);
    cudaFree(d_Z2_val); cudaFree(d_A2_val);
    cudaFree(d_Z3_val); cudaFree(d_A3_val);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total Training Time: %.2f ms\n", elapsedTime);

    cudaMemcpy(W1, d_W1, input_size * hidden1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1, d_b1, hidden1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2, d_W2, hidden1 * hidden2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2, d_b2, hidden2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W3, d_W3, hidden2 * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b3, d_b3, output_size * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_Z1); cudaFree(d_A1);
    cudaFree(d_Z2); cudaFree(d_A2);
    cudaFree(d_Z3); cudaFree(d_A3);
    cudaFree(d_dW1); cudaFree(d_db1);
    cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_dW3); cudaFree(d_db3);
    cudaFree(d_dZ3);
    cudaFree(d_dZ2);
    cudaFree(d_dZ1);
    cudaFree(d_A2_T);
    cudaFree(d_W3_T);
    cudaFree(d_A1_T);
    cudaFree(d_W2_T);
    cudaFree(d_X_T);
    
    float *Z1_final = (float*)malloc(test_samples * hidden1 * sizeof(float));
    float *A1_final = (float*)malloc(test_samples * hidden1 * sizeof(float));
    float *Z2_final = (float*)malloc(test_samples * hidden2 * sizeof(float));
    float *A2_final = (float*)malloc(test_samples * hidden2 * sizeof(float));
    float *Z3_final = (float*)malloc(test_samples * output_size * sizeof(float));
    float *A3_final = (float*)malloc(test_samples * output_size * sizeof(float));

    forward_propagation(X_test, W1, b1, W2, b2, W3, b3, test_samples, input_size, hidden1, hidden2, output_size, 
                        Z1_final, A1_final, Z2_final, A2_final, Z3_final, A3_final);

    float final_accuracy = compute_accuracy(A3_final, Y_test, test_samples, output_size);
    printf("Final Model Accuracy: %.4f\n", final_accuracy);
    printf("Grind rate: %f\n",60000*epochs/elapsedTime);

    free(Z1_final); free(A1_final); free(Z2_final); free(A2_final); free(Z3_final); free(A3_final);
    save_loss_to_binary(loss_history, epochs, "loss_history_gpu1.bin");
    free(loss_history);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
    cublasDestroy(handle);
}

void train(float* X_train, float* Y_train, float* X_val, float* Y_val, float* X_test, float* Y_test, 
            int train_samples, int val_samples, int test_samples, 
            int input_size, int hidden1, int hidden2, int output_size, int epochs, int batch_size, float learning_rate) {

    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    float *W1, *b1, *W2, *b2, *W3, *b3;
    initialize_weights(input_size, hidden1, hidden2, output_size, &W1, &b1, &W2, &b2, &W3, &b3);

    float* loss_history = (float*)malloc(epochs * sizeof(float));

    float *d_X_train, *d_Y_train, *d_X, *d_X_val, *d_Y, *d_W1, *d_b1, *d_W2, *d_b2, *d_W3, *d_b3;
    float *d_Z1, *d_A1, *d_Z2, *d_A2, *d_Z3, *d_A3;
    float *d_dW1, *d_db1, *d_dW2, *d_db2, *d_dW3, *d_db3;
    
    cudaMalloc(&d_X, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_X_val, val_samples * input_size * sizeof(float));
    cudaMalloc(&d_Y, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_W1, input_size * hidden1 * sizeof(float));
    cudaMalloc(&d_b1, hidden1 * sizeof(float));
    cudaMalloc(&d_W2, hidden1 * hidden2 * sizeof(float));
    cudaMalloc(&d_b2, hidden2 * sizeof(float));
    cudaMalloc(&d_W3, hidden2 * output_size * sizeof(float));
    cudaMalloc(&d_b3, output_size * sizeof(float));
    cudaMalloc(&d_Z1, batch_size * hidden1 * sizeof(float));
    cudaMalloc(&d_A1, batch_size * hidden1 * sizeof(float));
    cudaMalloc(&d_Z2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_A2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_Z3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_A3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_dW1, input_size * hidden1 * sizeof(float));
    cudaMalloc(&d_db1, hidden1 * sizeof(float));
    cudaMalloc(&d_dW2, hidden1 * hidden2 * sizeof(float));
    cudaMalloc(&d_db2, hidden2 * sizeof(float));
    cudaMalloc(&d_dW3, hidden2 * output_size * sizeof(float));
    cudaMalloc(&d_db3, output_size * sizeof(float));

    cudaMemcpy(d_X_val, X_val, val_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W1, W1, input_size * hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, b1, hidden1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, W2, hidden1 * hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, b2, hidden2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W3, W3, hidden2 * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b3, b3, output_size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_dZ3, *d_dZ2, *d_dZ1;
    cudaMalloc(&d_dZ3, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_dZ2, batch_size * hidden2 * sizeof(float));
    cudaMalloc(&d_dZ1, batch_size * hidden1 * sizeof(float));
    float* d_A2_T, * d_W3_T, * d_A1_T,* d_W2_T, * d_X_T;
    cudaMalloc(&d_A2_T, hidden2 * batch_size * sizeof(float));
    cudaMalloc(&d_W3_T, output_size * hidden2 * sizeof(float));
    cudaMalloc(&d_A1_T, hidden1 * batch_size * sizeof(float));
    cudaMalloc(&d_W2_T, hidden2 * hidden1 * sizeof(float));
    cudaMalloc(&d_X_T, input_size * batch_size * sizeof(float));

    float *d_Z1_val, *d_A1_val, *d_Z2_val, *d_A2_val, *d_Z3_val, *d_A3_val;
    cudaMalloc(&d_Z1_val, val_samples * hidden1 * sizeof(float));
    cudaMalloc(&d_A1_val, val_samples * hidden1 * sizeof(float));
    cudaMalloc(&d_Z2_val, val_samples * hidden2 * sizeof(float));
    cudaMalloc(&d_A2_val, val_samples * hidden2 * sizeof(float));
    cudaMalloc(&d_Z3_val, val_samples * output_size * sizeof(float));
    cudaMalloc(&d_A3_val, val_samples * output_size * sizeof(float));

    cudaMalloc(&d_X_train, train_samples * input_size * sizeof(float));
    cudaMalloc(&d_Y_train, train_samples * output_size * sizeof(float));
    cudaMemcpy(d_X_train, X_train, train_samples * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y_train, Y_train, train_samples * output_size * sizeof(float), cudaMemcpyHostToDevice);


    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_data(X_train, Y_train, train_samples, input_size, output_size);

        for (int i = 0; i < train_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > train_samples) ? (train_samples - i) : batch_size;

            float* d_X = d_X_train + i * input_size;
            float* d_Y = d_Y_train + i * output_size;
            forward_propagation_gpu(d_X, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                                    current_batch_size, input_size, hidden1, hidden2, output_size,
                                    d_Z1, d_A1, d_Z2, d_A2, d_Z3, d_A3);

            backward_propagation_gpu(d_X, d_Y, d_Z1, d_A1, d_Z2, d_A2, d_Z3, d_A3,
                                     d_W1, d_W2, d_W3, d_b1, d_b2, d_b3,
                                     current_batch_size, input_size, hidden1, hidden2, output_size,
                                     d_dW1, d_db1, d_dW2, d_db2, d_dW3, d_db3, d_dZ1, d_dZ2, d_dZ3,
                                     d_A2_T, d_W3_T, d_A1_T,d_W2_T, d_X_T,
                                     learning_rate);

            cudaDeviceSynchronize();
        }

        forward_propagation_gpu(d_X_val, d_W1, d_b1, d_W2, d_b2, d_W3, d_b3,
                                val_samples, input_size, hidden1, hidden2, output_size,
                                d_Z1_val, d_A1_val, d_Z2_val, d_A2_val, d_Z3_val, d_A3_val);

        float* A3_val = (float*)malloc(val_samples * output_size * sizeof(float));
        cudaMemcpy(A3_val, d_A3_val, val_samples * output_size * sizeof(float), cudaMemcpyDeviceToHost);
        loss_history[epoch] = compute_loss(A3_val, Y_val, val_samples, output_size);
        printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, loss_history[epoch]);

        free(A3_val);
    }
    cudaFree(d_Z1_val); cudaFree(d_A1_val);
    cudaFree(d_Z2_val); cudaFree(d_A2_val);
    cudaFree(d_Z3_val); cudaFree(d_A3_val);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total Training Time: %.2f ms\n", elapsedTime);

    cudaMemcpy(W1, d_W1, input_size * hidden1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b1, d_b1, hidden1 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W2, d_W2, hidden1 * hidden2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b2, d_b2, hidden2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(W3, d_W3, hidden2 * output_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b3, d_b3, output_size * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(d_X); cudaFree(d_Y);
    cudaFree(d_W1); cudaFree(d_b1);
    cudaFree(d_W2); cudaFree(d_b2);
    cudaFree(d_W3); cudaFree(d_b3);
    cudaFree(d_Z1); cudaFree(d_A1);
    cudaFree(d_Z2); cudaFree(d_A2);
    cudaFree(d_Z3); cudaFree(d_A3);
    cudaFree(d_dW1); cudaFree(d_db1);
    cudaFree(d_dW2); cudaFree(d_db2);
    cudaFree(d_dW3); cudaFree(d_db3);
    cudaFree(d_dZ3);
    cudaFree(d_dZ2);
    cudaFree(d_dZ1);
    cudaFree(d_A2_T);
    cudaFree(d_W3_T);
    cudaFree(d_A1_T);
    cudaFree(d_W2_T);
    cudaFree(d_X_T);
    
    float *Z1_final = (float*)malloc(test_samples * hidden1 * sizeof(float));
    float *A1_final = (float*)malloc(test_samples * hidden1 * sizeof(float));
    float *Z2_final = (float*)malloc(test_samples * hidden2 * sizeof(float));
    float *A2_final = (float*)malloc(test_samples * hidden2 * sizeof(float));
    float *Z3_final = (float*)malloc(test_samples * output_size * sizeof(float));
    float *A3_final = (float*)malloc(test_samples * output_size * sizeof(float));

    forward_propagation(X_test, W1, b1, W2, b2, W3, b3, test_samples, input_size, hidden1, hidden2, output_size, 
                        Z1_final, A1_final, Z2_final, A2_final, Z3_final, A3_final);

    float final_accuracy = compute_accuracy(A3_final, Y_test, test_samples, output_size);
    printf("Final Model Accuracy: %.4f\n", final_accuracy);
    printf("Grind rate: %f\n",60000*epochs/elapsedTime);

    free(Z1_final); free(A1_final); free(Z2_final); free(A2_final); free(Z3_final); free(A3_final);
    save_loss_to_binary(loss_history, epochs, "loss_history_gpu0.bin");
    free(loss_history);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
}


int main(int argc, char *argv[]) {
    int use_blas = atoi(argv[1]);
    // Params
    int hidden1 = 128;
    int hidden2 = 256;
    int output_size = 10;
    int epochs = 50;
    int batch_size = 500;
    float learning_rate = 0.01;

    srand(42);

    int total_samples, test_samples;
    int input_size = 28 * 28;
    float *X_full = load_mnist_images("train-images-idx3-ubyte.gz", &total_samples);
    float *Y_full = load_mnist_labels("train-labels-idx1-ubyte.gz", &total_samples);
    float *X_test = load_mnist_images("t10k-images-idx3-ubyte.gz", &test_samples);
    float *Y_test = load_mnist_labels("t10k-labels-idx1-ubyte.gz", &test_samples);

    int train_samples = 50000;
    int val_samples = 10000;

    float *X_train = X_full;
    float *Y_train = Y_full;
    float *X_val = X_full + (train_samples * input_size);
    float *Y_val = Y_full + (train_samples * output_size);

    printf("Epochs: %d, Batch size: %d, Learning rate: %f \n",epochs ,batch_size, learning_rate);

    if (use_blas) {
        printf("Using BLAS implementation.\n");
        train_b(X_train, Y_train, X_val, Y_val, X_test, Y_test, train_samples, val_samples, test_samples,
                input_size, hidden1, hidden2, output_size, epochs, batch_size, learning_rate);
    } else {
        printf("Using hand-optimized implementation.\n");
        train(X_train, Y_train, X_val, Y_val, X_test, Y_test, train_samples, val_samples, test_samples,
              input_size, hidden1, hidden2, output_size, epochs, batch_size, learning_rate);
    }

    free(X_full);
    free(Y_full);
    free(X_test);
    free(Y_test);
    return 0;
}
