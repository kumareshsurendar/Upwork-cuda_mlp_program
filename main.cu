#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

// Forward declarations of kernels
__global__ void reluForward(float *input, float *output, int size);
__global__ void sigmoidBackward(float *output, float *gradOutput, float *gradInput, int size);
__global__ void linearForward(float *input, float *weights, float *bias, float *output, int inputSize, int outputSize);

float computeAccuracy(const std::vector<float>& predictions, const std::vector<float>& targets) {
    int correct = 0;
    int total = predictions.size();
    for (int i = 0; i < total; ++i) {
        int pred_label = predictions[i] >= 0.5f ? 1 : 0;
        if (pred_label == static_cast<int>(targets[i])) {
            correct++;
        }
    }
    return static_cast<float>(correct) / total;
}

// Helper to generate labeled data points based on quadrant rule
void generateData(std::vector<float>& data, std::vector<float>& labels, int samples) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    data.resize(samples * 2);
    labels.resize(samples);

    for (int i = 0; i < samples; ++i) {
        float x = dis(gen);
        float y = dis(gen);
        data[i * 2] = x;
        data[i * 2 + 1] = y;
        labels[i] = ((x > 0 && y > 0) || (x < 0 && y < 0)) ? 1.0f : 0.0f;
    }
}

int main() {
    const int inputSize = 2;
    const int hiddenSize = 4;
    const int outputSize = 1;
    const int samples = 1024;

    std::vector<float> hostInput, hostLabels;
    generateData(hostInput, hostLabels, samples);

    // Allocate device memory
    float *d_input, *d_hidden, *d_output, *d_weights1, *d_bias1, *d_weights2, *d_bias2;
    cudaMalloc(&d_input, samples * inputSize * sizeof(float));
    cudaMalloc(&d_hidden, samples * hiddenSize * sizeof(float));
    cudaMalloc(&d_output, samples * outputSize * sizeof(float));
    cudaMalloc(&d_weights1, hiddenSize * inputSize * sizeof(float));
    cudaMalloc(&d_bias1, hiddenSize * sizeof(float));
    cudaMalloc(&d_weights2, outputSize * hiddenSize * sizeof(float));
    cudaMalloc(&d_bias2, outputSize * sizeof(float));

    // Initialize weights and biases (for demo, zeros)
    cudaMemset(d_weights1, 0, hiddenSize * inputSize * sizeof(float));
    cudaMemset(d_bias1, 0, hiddenSize * sizeof(float));
    cudaMemset(d_weights2, 0, outputSize * hiddenSize * sizeof(float));
    cudaMemset(d_bias2, 0, outputSize * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, hostInput.data(), samples * inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // Run forward pass
    dim3 blockDim(256);
    dim3 gridDim1((samples * hiddenSize + blockDim.x - 1) / blockDim.x);
    dim3 gridDim2((samples * outputSize + blockDim.x - 1) / blockDim.x);

    for (int i = 0; i < samples; ++i) {
        float* sampleInput = d_input + i * inputSize;
        float* sampleHidden = d_hidden + i * hiddenSize;
        float* sampleOutput = d_output + i * outputSize;

        linearForward<<<1, hiddenSize>>>(sampleInput, d_weights1, d_bias1, sampleHidden, inputSize, hiddenSize);
        reluForward<<<1, hiddenSize>>>(sampleHidden, sampleHidden, hiddenSize);
        linearForward<<<1, outputSize>>>(sampleHidden, d_weights2, d_bias2, sampleOutput, hiddenSize, outputSize);
    }

    // Copy output to host
    std::vector<float> predictions(samples);
    cudaMemcpy(predictions.data(), d_output, samples * sizeof(float), cudaMemcpyDeviceToHost);

    // Evaluate accuracy
    float accuracy = computeAccuracy(predictions, hostLabels);
    std::cout << "Network accuracy: " << accuracy << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_weights1);
    cudaFree(d_bias1);
    cudaFree(d_weights2);
    cudaFree(d_bias2);

    return 0;
}
