#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "coordinates_dataset.h"
#include "bce_cost.h"
#include "matrix.h"

// Forward declarations of kernels
__global__ void reluForward(float *input, float *output, int size);
__global__ void sigmoidBackward(float *output, float *gradOutput, float *gradInput, int size);
__global__ void linearForward(float *input, float *weights, float *bias, float *output, int inputSize, int outputSize);


float computeAccuracy(float* predictions, float* targets, int count) {
    int correct = 0;
    for (int i = 0; i < count; ++i) {
        int pred_label = predictions[i] >= 0.5f ? 1 : 0;
        if (pred_label == static_cast<int>(targets[i])) {
            correct++;
        }
    }
    return static_cast<float>(correct) / count;
}

int main() {
    srand(time(NULL));

    const int samples = 1024;
    const int inputSize = 2;
    const int hiddenSize = 4;
    const int outputSize = 1;

    CoordinatesDataset dataset(samples, 1);
    Matrix& inputMatrix = dataset.getBatches()[0];
    Matrix& targetMatrix = dataset.getTargets()[0];
    inputMatrix.copyDeviceToHost();
    targetMatrix.copyDeviceToHost();

    float* hostInput = inputMatrix.data_host.get();
    float* hostLabels = targetMatrix.data_host.get();

    float *d_input, *d_hidden, *d_output, *d_weights1, *d_bias1, *d_weights2, *d_bias2;
    cudaMalloc(&d_input, samples * inputSize * sizeof(float));
    cudaMalloc(&d_hidden, samples * hiddenSize * sizeof(float));
    cudaMalloc(&d_output, samples * outputSize * sizeof(float));
    cudaMalloc(&d_weights1, hiddenSize * inputSize * sizeof(float));
    cudaMalloc(&d_bias1, hiddenSize * sizeof(float));
    cudaMalloc(&d_weights2, outputSize * hiddenSize * sizeof(float));
    cudaMalloc(&d_bias2, outputSize * sizeof(float));

    cudaMemset(d_weights1, 0, hiddenSize * inputSize * sizeof(float));
    cudaMemset(d_bias1, 0, hiddenSize * sizeof(float));
    cudaMemset(d_weights2, 0, outputSize * hiddenSize * sizeof(float));
    cudaMemset(d_bias2, 0, outputSize * sizeof(float));

    cudaMemcpy(d_input, hostInput, samples * inputSize * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < samples; ++i) {
        float* sampleInput = d_input + i * inputSize;
        float* sampleHidden = d_hidden + i * hiddenSize;
        float* sampleOutput = d_output + i * outputSize;

        linearForward<<<1, hiddenSize>>>(sampleInput, d_weights1, d_bias1, sampleHidden, inputSize, hiddenSize);
        reluForward<<<1, hiddenSize>>>(sampleHidden, sampleHidden, hiddenSize);
        linearForward<<<1, outputSize>>>(sampleHidden, d_weights2, d_bias2, sampleOutput, hiddenSize, outputSize);
    }

    std::vector<float> predictions(samples);
    cudaMemcpy(predictions.data(), d_output, samples * sizeof(float), cudaMemcpyDeviceToHost);

    float accuracy = computeAccuracy(predictions.data(), hostLabels, samples);
    std::cout << "Network accuracy: " << accuracy << std::endl;

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);

    return 0;
}
