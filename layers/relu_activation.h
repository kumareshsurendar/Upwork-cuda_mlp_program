#pragma once

#include "nn_layer.h"

class ReLUActivation : public NNLayer {
private:
	Matrix A;

	Matrix Z;
	Matrix dZ;

public:
	ReLUActivation(std::string name);
	~ReLUActivation();

	/// <summary>
	/// Calculates max(Z, 0)
	/// </summary>
	/// <param name="Z">Layer input</param>
	/// <returns>ReLu(Z)</returns>
	Matrix& forward(Matrix& Z);
	
	/// <summary>
	/// dZ = dA if Z > 0; 0 otherwise
	/// </summary>
	/// <param name="dA">Error from the previous layer</param>
	/// <param name="learning_rate">The learning rate. Does not have effect here</param>
	/// <returns>The error gradient</returns>
	Matrix& backprop(Matrix& dA, float learning_rate = 0.01);
};
