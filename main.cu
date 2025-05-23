#include <iostream>
#include "matrix.h"
#include "coordinates_dataset.cuh"
#include "neural_network.cuh"
#include "bce_cost.cuh"

float computeAccuracy(const Matrix& predictions, const Matrix& targets) {
    int correct = 0;
    for (int i = 0; i < predictions.rows; ++i) {
        int predicted_label = predictions.data[i] >= 0.5f ? 1 : 0;
        int target_label = targets.data[i] >= 0.5f ? 1 : 0;
        if (predicted_label == target_label) {
            correct++;
        }
    }
    return static_cast<float>(correct) / predictions.rows;
}

int main() {
    srand( time(NULL) );

    CoordinatesDataset dataset(100, 21);
    BCECost bce_cost;

    NeuralNetwork nn;
    nn.addLayer(new LinearLayer("linear_1", Shape(2, 30)));
    nn.addLayer(new ReLUActivation("relu_1"));
    nn.addLayer(new LinearLayer("linear_2", Shape(30, 1)));
    nn.addLayer(new SigmoidActivation("sigmoid_output"));

    // network training
    Matrix Y;
    for (int epoch = 0; epoch < 1001; epoch++) {
        float cost = 0.0;

        for (int batch = 0; batch < dataset.getNumOfBatches() - 1; batch++) {
            Y = nn.forward(dataset.getBatches().at(batch));
            nn.backprop(Y, dataset.getTargets().at(batch));
            cost += bce_cost.cost(Y, dataset.getTargets().at(batch));
        }

        if (epoch % 100 == 0) {
            std::cout  << "Epoch: " << epoch
                      << ", Cost: " << cost / dataset.getNumOfBatches()
                      << std::endl;
        }
    }

    // compute accuracy
    Y = nn.forward(dataset.getBatches().at(dataset.getNumOfBatches() - 1));
    Y.copyDeviceToHost();

    float accuracy = computeAccuracy(
            Y, dataset.getTargets().at(dataset.getNumOfBatches() - 1));
    std::cout  << "Accuracy: " << accuracy << std::endl;

    return 0;
}
