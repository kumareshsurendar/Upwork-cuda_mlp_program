#pragma once

#include <vector>
#include "layers/nn_layer.h"
#include "nn_utils/bce_cost.h"

class NeuralNetwork {
private:
	std::vector<NNLayer*> layers;
	BCECost bce_cost;

	Matrix Y;
	Matrix dY;
	float learning_rate;

public:
	NeuralNetwork(float learning_rate = 0.01);
	~NeuralNetwork();

	/// <summary>
	/// Calculates forward operation on all layers
	/// </summary>
	/// <param name="X">The network input</param>
	/// <returns>Output of the last layer</returns>
	Matrix forward(Matrix X);
	
	/// <summary>
	/// Performs backward operation on all layers
	/// </summary>
	/// <param name="predictions">The actual ouptut of the network</param>
	/// <param name="target">The desired output of the network</param>
	void backprop(Matrix predictions, Matrix target);

	/// <summary>
	/// Adds a layer to the network
	/// </summary>
	/// <param name="layer">The layer to be added</param>
	void addLayer(NNLayer *layer);
	
	std::vector<NNLayer*> getLayers() const;

};
