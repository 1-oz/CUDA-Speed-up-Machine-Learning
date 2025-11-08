#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <zlib.h> 
#include <omp.h>
#include <cblas.h>

#define IMAGE_SIZE 28 * 28
#define NUM_CLASSES 10

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
void relu1(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0.0f, x[i]);
    }
}

void relu(float* output, float* input, int size) {
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        output[i] = fmaxf(0.0f, input[i]);
    }
}

// ReLU Derivative Func
void relu_derivative(float* x, float* output, int size) {
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        output[i] = (x[i] > 0) ? 1.0f : 0.0f;
    }
}

//Softmax
void softmax(float* x, float* output, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum_exp = 0.0f;
        #pragma omp parallel for reduction(+:sum_exp)
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(x[i * cols + j]);
            sum_exp += output[i * cols + j];
        }
        #pragma omp parallel for
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] /= sum_exp;
        }
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

    float std_W1 = sqrtf(2.0f / input_size);
    float std_W2 = sqrtf(2.0f / hidden1);
    float std_W3 = sqrtf(2.0f / hidden2);

    for (int i = 0; i < input_size * hidden1; i++) (*W1)[i] = std_W1 * randn();
    for (int i = 0; i < hidden1 * hidden2; i++) (*W2)[i] = std_W2 * randn();
    for (int i = 0; i < hidden2 * output_size; i++) (*W3)[i] = std_W3 * randn();
}


//tiled matmul 
void matmul(float* A, float* B, float* C, int m, int n, int p) {
    memset(C, 0, m * p * sizeof(float));
    int block_size = 2;
    for (int i = 0; i < m; i += block_size) {
        for (int j = 0; j < p; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int bi = 0; bi < block_size && (i + bi) < m; bi++) {
                    for (int bj = 0; bj < block_size && (j + bj) < p; bj++) {
                        float sum = C[(i + bi) * p + (j + bj)];
                        
                        for (int bk = 0; bk < block_size && (k + bk) < n; bk++) {
                            sum += A[(i + bi) * n + (k + bk)] * B[(k + bk) * p + (j + bj)];
                        }

                        C[(i + bi) * p + (j + bj)] = sum;
                    }
                }
            }
        }
    }
}
void add_bias(float* A, float* b, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] += b[j];
        }
    }
}

//hand-coded GEMM
void gemm(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    memset(C, 0, M * N * sizeof(float));
    int block_size = 4;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i += block_size){
        for (int j = 0; j < N; j += block_size){
            for (int k = 0; k < K; k += block_size){
                for (int bi = 0; bi < block_size && (i+bi)<M;bi++){
                    for (int bj = 0; bj < block_size && (j+bj)<M;bj++){
                        float sum = 0.0f;

                        for (int bk = 0; bk < block_size && (k + bk) < K; bk++){
                            sum += A[(i + bi) * K + (k + bk)] * B[(k+bk)*N + (j+bj)];
                        }
                        C[(i+bi)*N + (j+bj)] += sum;
                    }
                }
            }
        }
    }
    if (bias != NULL){
        #pragma omp parallel for
        for (int i = 0; i < M; i++){
            for (int j = 0; j < N; j++){
                C[i*N + j] += bias[j];
            }
        }
    }
}

//cblas GEMM
void gemm_openblas(float* A, float* B, float* C, float* bias, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 
                1.0f, A, K, B, N, 0.0f, C, N);

    if (bias != NULL) {
        #pragma omp parallel for
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * N + j] += bias[j];
            }
        }
    }
}

void forward_propagation(float* X, float* W1, float* b1, float* W2, float* b2, 
                         float* W3, float* b3, int batch_size, int input_size, 
                         int hidden1, int hidden2, int output_size,
                         float* Z1, float* A1, float* Z2, float* A2, float* Z3, float* A3) {
    
    // Z1 = X * W1 + b1
    gemm(X, W1, Z1, b1, batch_size, hidden1, input_size);
    relu(A1, Z1, batch_size * hidden1);

    // Z2 = A1 * W2 + b2
    gemm(A1, W2, Z2, b2, batch_size, hidden2, hidden1);
    relu(A2, Z2, batch_size * hidden2);

    // Z3 = A2 * W3 + b3
    gemm(A2, W3, Z3, b3, batch_size, output_size, hidden2);

    // A3 = Softmax(Z3)
    softmax(Z3, A3, batch_size, output_size);
}
//cblas version forward propa
void forward_propagation_b(float* X, float* W1, float* b1, float* W2, float* b2, 
                         float* W3, float* b3, int batch_size, int input_size, 
                         int hidden1, int hidden2, int output_size,
                         float* Z1, float* A1, float* Z2, float* A2, float* Z3, float* A3) {
    
    // Z1 = X * W1 + b1
    gemm_openblas(X, W1, Z1, b1, batch_size, hidden1, input_size);
    relu(A1, Z1, batch_size * hidden1);  // A1 = ReLU(Z1)

    // Z2 = A1 * W2 + b2
    gemm_openblas(A1, W2, Z2, b2, batch_size, hidden2, hidden1);
    relu(A2, Z2, batch_size * hidden2);  // A2 = ReLU(Z2)

    // Z3 = A2 * W3 + b3
    gemm_openblas(A2, W3, Z3, b3, batch_size, output_size, hidden2);

    // A3 = Softmax(Z3)
    softmax(Z3, A3, batch_size, output_size);
}


//Mat transpose
void transpose(float* A, float* B, int m, int n) {
    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        #pragma omp simd
        for (int i = 0; i < m; i++) {
            B[j * m + i] = A[i * n + j];
        }
    }
}


void elementwise_multiply(float* A, float* B, float* C, int size) {
    #pragma omp parallel for simd
    for (int i = 0; i < size; i++) {
        C[i] = A[i] * B[i];
    }
}

void column_sum(float* A, float* result, int m, int n) {
    memset(result, 0, n * sizeof(float));

    #pragma omp parallel for
    for (int j = 0; j < n; j++) {
        float sum = 0.0f;

        #pragma omp simd reduction(+:sum)
        for (int i = 0; i < m; i++) {
            sum += A[i * n + j];
        }

        result[j] = sum / m;
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
    gemm(A2_T, dZ3, dW3, NULL, hidden2, output_size, batch_size);
    for (int i = 0; i < hidden2 * output_size; i++) dW3[i] /= batch_size;
    free(A2_T);

    // db3 = sum(dZ3, axis=0) / batch_size
    column_sum(dZ3, db3, batch_size, output_size);

    // dZ2 = (dZ3 * W3^T) ⊙ ReLU'(Z2)
    float* W3_T = (float*)malloc(output_size * hidden2 * sizeof(float));
    transpose(W3, W3_T, hidden2, output_size);
    float* dZ2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    gemm(dZ3, W3_T, dZ2, NULL, batch_size, hidden2, output_size);
    float* dReLU_Z2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    relu_derivative(Z2, dReLU_Z2, batch_size * hidden2);
    elementwise_multiply(dZ2, dReLU_Z2, dZ2, batch_size * hidden2);  // ⊙ 乘法
    free(W3_T);
    free(dZ3);
    free(dReLU_Z2);

    // dW2 = (A1^T * dZ2) / batch_size
    float* A1_T = (float*)malloc(hidden1 * batch_size * sizeof(float));
    transpose(A1, A1_T, batch_size, hidden1);
    gemm(A1_T, dZ2, dW2, NULL, hidden1, hidden2, batch_size);
    for (int i = 0; i < hidden1 * hidden2; i++) dW2[i] /= batch_size;
    free(A1_T);

    // db2 = sum(dZ2, axis=0) / batch_size
    column_sum(dZ2, db2, batch_size, hidden2);

    // dZ1 = (dZ2 * W2^T) ⊙ ReLU'(Z1)
    float* W2_T = (float*)malloc(hidden2 * hidden1 * sizeof(float));
    transpose(W2, W2_T, hidden1, hidden2);
    float* dZ1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    gemm(dZ2, W2_T, dZ1, NULL, batch_size, hidden1, hidden2);
    float* dReLU_Z1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    relu_derivative(Z1, dReLU_Z1, batch_size * hidden1);
    elementwise_multiply(dZ1, dReLU_Z1, dZ1, batch_size * hidden1);
    free(W2_T);
    free(dZ2);
    free(dReLU_Z1);

    // dW1 = (X^T * dZ1) / batch_size
    float* X_T = (float*)malloc(input_size * batch_size * sizeof(float));
    transpose(X, X_T, batch_size, input_size);
    gemm(X_T, dZ1, dW1, NULL, input_size, hidden1, batch_size);
    for (int i = 0; i < input_size * hidden1; i++) dW1[i] /= batch_size;
    free(X_T);

    // db1 = sum(dZ1, axis=0) / batch_size
    column_sum(dZ1, db1, batch_size, hidden1);

    free(dZ1);
}
//cblas version forward propa
void backward_propagation_b(float* X, float* Y, float* Z1, float* A1, float* Z2, float* A2, 
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
    gemm_openblas(A2_T, dZ3, dW3, NULL, hidden2, output_size, batch_size);
    for (int i = 0; i < hidden2 * output_size; i++) dW3[i] /= batch_size;
    free(A2_T);

    // db3 = sum(dZ3, axis=0) / batch_size
    column_sum(dZ3, db3, batch_size, output_size);

    // dZ2 = (dZ3 * W3^T) ⊙ ReLU'(Z2)
    float* W3_T = (float*)malloc(output_size * hidden2 * sizeof(float));
    transpose(W3, W3_T, hidden2, output_size);
    float* dZ2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    gemm_openblas(dZ3, W3_T, dZ2, NULL, batch_size, hidden2, output_size);
    float* dReLU_Z2 = (float*)malloc(batch_size * hidden2 * sizeof(float));
    relu_derivative(Z2, dReLU_Z2, batch_size * hidden2);
    elementwise_multiply(dZ2, dReLU_Z2, dZ2, batch_size * hidden2);
    free(W3_T);
    free(dZ3);
    free(dReLU_Z2);

    // dW2 = (A1^T * dZ2) / batch_size
    float* A1_T = (float*)malloc(hidden1 * batch_size * sizeof(float));
    transpose(A1, A1_T, batch_size, hidden1);
    gemm_openblas(A1_T, dZ2, dW2, NULL, hidden1, hidden2, batch_size);
    for (int i = 0; i < hidden1 * hidden2; i++) dW2[i] /= batch_size;
    free(A1_T);

    // db2 = sum(dZ2, axis=0) / batch_size
    column_sum(dZ2, db2, batch_size, hidden2);

    // dZ1 = (dZ2 * W2^T) ⊙ ReLU'(Z1)
    float* W2_T = (float*)malloc(hidden2 * hidden1 * sizeof(float));
    transpose(W2, W2_T, hidden1, hidden2);
    float* dZ1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    gemm_openblas(dZ2, W2_T, dZ1, NULL, batch_size, hidden1, hidden2);
    float* dReLU_Z1 = (float*)malloc(batch_size * hidden1 * sizeof(float));
    relu_derivative(Z1, dReLU_Z1, batch_size * hidden1);
    elementwise_multiply(dZ1, dReLU_Z1, dZ1, batch_size * hidden1);
    free(W2_T);
    free(dZ2);
    free(dReLU_Z1);

    // dW1 = (X^T * dZ1) / batch_size
    float* X_T = (float*)malloc(input_size * batch_size * sizeof(float));
    transpose(X, X_T, batch_size, input_size);
    gemm_openblas(X_T, dZ1, dW1, NULL, input_size, hidden1, batch_size);
    for (int i = 0; i < input_size * hidden1; i++) dW1[i] /= batch_size;
    free(X_T);

    // db1 = sum(dZ1, axis=0) / batch_size
    column_sum(dZ1, db1, batch_size, hidden1);

    free(dZ1);
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

void save_loss_to_binary(float* loss, int epochs, const char* filename) {
    FILE* file = fopen(filename, "wb");
    fwrite(loss, sizeof(float), epochs, file);
    fclose(file);
}

float compute_loss(float* A3, float* Y, int samples, int output_size) {
    float loss = 0.0f;
    #pragma omp parallel for reduction(+:loss)
    for (int i = 0; i < samples; i++) {
        for (int j = 0; j < output_size; j++) {
            if (Y[i * output_size + j] > 0) {
                loss += -logf(A3[i * output_size + j] + 1e-9);
            }
        }
    }
    return loss / samples;
}

void train(float* X_train, float* Y_train, float* X_val, float* Y_val, float* X_test, float* Y_test, 
            int train_samples, int val_samples, int test_samples, 
            int input_size, int hidden1, int hidden2, int output_size, int epochs, int batch_size, float learning_rate) {

    double train_start = omp_get_wtime();
    float *W1, *b1, *W2, *b2, *W3, *b3;
    initialize_weights(input_size, hidden1, hidden2, output_size, &W1, &b1, &W2, &b2, &W3, &b3);

    float* loss_history = (float*)malloc(epochs * sizeof(float));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_data(X_train, Y_train, train_samples, input_size, output_size);

        for (int i = 0; i < train_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > train_samples) ? (train_samples - i) : batch_size;

            float *Z1 = (float*)malloc(current_batch_size * hidden1 * sizeof(float));
            float *A1 = (float*)malloc(current_batch_size * hidden1 * sizeof(float));
            float *Z2 = (float*)malloc(current_batch_size * hidden2 * sizeof(float));
            float *A2 = (float*)malloc(current_batch_size * hidden2 * sizeof(float));
            float *Z3 = (float*)malloc(current_batch_size * output_size * sizeof(float));
            float *A3 = (float*)malloc(current_batch_size * output_size * sizeof(float));

            float *dW1 = (float*)malloc(input_size * hidden1 * sizeof(float));
            float *db1 = (float*)malloc(hidden1 * sizeof(float));
            float *dW2 = (float*)malloc(hidden1 * hidden2 * sizeof(float));
            float *db2 = (float*)malloc(hidden2 * sizeof(float));
            float *dW3 = (float*)malloc(hidden2 * output_size * sizeof(float));
            float *db3 = (float*)malloc(output_size * sizeof(float));

            forward_propagation(&X_train[i * input_size], W1, b1, W2, b2, W3, b3, current_batch_size, input_size, hidden1, hidden2, output_size, Z1, A1, Z2, A2, Z3, A3);
            
            backward_propagation(&X_train[i * input_size], &Y_train[i * output_size], Z1, A1, Z2, A2, Z3, A3, W1, W2, W3,
                                 current_batch_size, input_size, hidden1, hidden2, output_size, dW1, db1, dW2, db2, dW3, db3);

            update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, input_size, hidden1, hidden2, output_size, learning_rate);

            free(Z1); free(A1); free(Z2); free(A2); free(Z3); free(A3);
            free(dW1); free(db1); free(dW2); free(db2); free(dW3); free(db3);
        }

        // In-training evaluation
        float *Z1_val = (float*)malloc(val_samples * hidden1 * sizeof(float));
        float *A1_val = (float*)malloc(val_samples * hidden1 * sizeof(float));
        float *Z2_val = (float*)malloc(val_samples * hidden2 * sizeof(float));
        float *A2_val = (float*)malloc(val_samples * hidden2 * sizeof(float));
        float *Z3_val = (float*)malloc(val_samples * output_size * sizeof(float));
        float *A3_val = (float*)malloc(val_samples * output_size * sizeof(float));

        forward_propagation(X_val, W1, b1, W2, b2, W3, b3, val_samples, input_size, hidden1, hidden2, output_size, 
                            Z1_val, A1_val, Z2_val, A2_val, Z3_val, A3_val);
        
        loss_history[epoch] = compute_loss(A3_val, Y_val, val_samples, output_size);
        printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, loss_history[epoch]);

        free(Z1_val); free(A1_val); free(Z2_val); free(A2_val); free(Z3_val); free(A3_val);
    }
    double train_end = omp_get_wtime();
    printf("Total Training Time: %f seconds\n", train_end - train_start);
    
    double inf_start = omp_get_wtime();
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
    printf("Grind rate: %f\n",60000*epochs/(train_end - train_start));
    double inf_end = omp_get_wtime();
    printf("Inference Time: %f seconds\n", inf_end - inf_start);

    free(Z1_final); free(A1_final); free(Z2_final); free(A2_final); free(Z3_final); free(A3_final);
    save_loss_to_binary(loss_history, epochs, "loss_history_0.bin");
    free(loss_history);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
}

void train_b(float* X_train, float* Y_train, float* X_val, float* Y_val, float* X_test, float* Y_test, 
            int train_samples, int val_samples, int test_samples, 
            int input_size, int hidden1, int hidden2, int output_size, int epochs, int batch_size, float learning_rate) {

    double train_start = omp_get_wtime();
    float *W1, *b1, *W2, *b2, *W3, *b3;
    initialize_weights(input_size, hidden1, hidden2, output_size, &W1, &b1, &W2, &b2, &W3, &b3);
    float* loss_history = (float*)malloc(epochs * sizeof(float));
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        shuffle_data(X_train, Y_train, train_samples, input_size, output_size);
        for (int i = 0; i < train_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > train_samples) ? (train_samples - i) : batch_size;

            float *Z1 = (float*)malloc(current_batch_size * hidden1 * sizeof(float));
            float *A1 = (float*)malloc(current_batch_size * hidden1 * sizeof(float));
            float *Z2 = (float*)malloc(current_batch_size * hidden2 * sizeof(float));
            float *A2 = (float*)malloc(current_batch_size * hidden2 * sizeof(float));
            float *Z3 = (float*)malloc(current_batch_size * output_size * sizeof(float));
            float *A3 = (float*)malloc(current_batch_size * output_size * sizeof(float));

            float *dW1 = (float*)malloc(input_size * hidden1 * sizeof(float));
            float *db1 = (float*)malloc(hidden1 * sizeof(float));
            float *dW2 = (float*)malloc(hidden1 * hidden2 * sizeof(float));
            float *db2 = (float*)malloc(hidden2 * sizeof(float));
            float *dW3 = (float*)malloc(hidden2 * output_size * sizeof(float));
            float *db3 = (float*)malloc(output_size * sizeof(float));

            forward_propagation_b(&X_train[i * input_size], W1, b1, W2, b2, W3, b3, current_batch_size, input_size, hidden1, hidden2, output_size, Z1, A1, Z2, A2, Z3, A3);
            
            backward_propagation_b(&X_train[i * input_size], &Y_train[i * output_size], Z1, A1, Z2, A2, Z3, A3, W1, W2, W3,
                                 current_batch_size, input_size, hidden1, hidden2, output_size, dW1, db1, dW2, db2, dW3, db3);

            update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, input_size, hidden1, hidden2, output_size, learning_rate);

            free(Z1); free(A1); free(Z2); free(A2); free(Z3); free(A3);
            free(dW1); free(db1); free(dW2); free(db2); free(dW3); free(db3);
        }

        // In-training evaluation
        float *Z1_val = (float*)malloc(val_samples * hidden1 * sizeof(float));
        float *A1_val = (float*)malloc(val_samples * hidden1 * sizeof(float));
        float *Z2_val = (float*)malloc(val_samples * hidden2 * sizeof(float));
        float *A2_val = (float*)malloc(val_samples * hidden2 * sizeof(float));
        float *Z3_val = (float*)malloc(val_samples * output_size * sizeof(float));
        float *A3_val = (float*)malloc(val_samples * output_size * sizeof(float));

        forward_propagation(X_val, W1, b1, W2, b2, W3, b3, val_samples, input_size, hidden1, hidden2, output_size, 
                            Z1_val, A1_val, Z2_val, A2_val, Z3_val, A3_val);
        
        loss_history[epoch] = compute_loss(A3_val, Y_val, val_samples, output_size);
        printf("Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, loss_history[epoch]);

        free(Z1_val); free(A1_val); free(Z2_val); free(A2_val); free(Z3_val); free(A3_val);
    }
    double train_end = omp_get_wtime();
    printf("Total Training Time(blas): %f seconds\n", train_end - train_start);
    
    double inf_start = omp_get_wtime();
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
    printf("Grind rate: %f\n",60000*epochs/(train_end - train_start));
    double inf_end = omp_get_wtime();
    printf("Inference Time: %f seconds\n", inf_end - inf_start);

    free(Z1_final); free(A1_final); free(Z2_final); free(A2_final); free(Z3_final); free(A3_final);
    save_loss_to_binary(loss_history, epochs, "loss_history_1.bin");
    free(loss_history);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
}




int main(int argc, char *argv[]) {
    int num_threads = 32;
    omp_set_num_threads(num_threads);

    int use_blas = atoi(argv[1]);

    // Params
    int hidden1 = 128;
    int hidden2 = 256;
    int output_size = 10;
    int epochs = 50;
    int batch_size = 500;
    float learning_rate = 0.01;

    srand(42);

    int total_samples;
    int test_samples;
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

    printf("Epochs: %d, Batch size: %d, Learning rate: %f\n", epochs, batch_size, learning_rate);
    printf("Training samples: %d, Validation samples: %d, Test samples: %d\n",
           train_samples, val_samples, test_samples);

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


