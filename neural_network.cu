// === Neural Network Kernel Implementations ===

#include <cmath>
#include <vector>

// ReLUActivation forward kernel
__global__ void reluForward(float *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLUActivation backward kernel
__global__ void reluBackward(float *input, float *gradOutput, float *gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gradInput[idx] = input[idx] > 0 ? gradOutput[idx] : 0.0f;
    }
}

// SigmoidActivation backward kernel
__global__ void sigmoidBackward(float *output, float *gradOutput, float *gradInput, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid_val = output[idx];
        gradInput[idx] = gradOutput[idx] * sigmoid_val * (1.0f - sigmoid_val);
    }
}

// LinearLayer forward kernel
__global__ void linearForward(float *input, float *weights, float *bias, float *output, int inputSize, int outputSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < outputSize) {
        float sum = 0.0f;
        for (int col = 0; col < inputSize; col++) {
            sum += weights[row * inputSize + col] * input[col];
        }
        output[row] = sum + bias[row];
    }
}

// computeAccuracy function (usually in main.cpp)
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
