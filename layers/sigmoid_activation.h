#pragma once

#include "nn_layer.h"

class SigmoidActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:

	SigmoidActivation(std::string name);	
	~SigmoidActivation();

	/// <summary>
	/// A = exp(Z) / (1 + exp(Z))
	/// </summary>
	/// <param name="Z">Input of the layer</param>
	/// <returns>Sigmoid(Z)</returns>
	Matrix& forward(Matrix& Z);
	
	/// <summary>
	/// Calculates the derivative. dZ = dA * sigmoid(Z) * (1 - sigmoid(Z))
	/// </summary>
	/// <param name="dA">The error on the output of this layer.</param>
	/// <param name="learning_rate">The learning rate.</param>
	/// <returns></returns>
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
