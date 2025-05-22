#pragma once
#include "nn_layer.h"

class LinearLayer : public NNLayer {
private:
	const float weights_init_threshold = 0.01;

	Matrix W;
	Matrix b;

	Matrix Z;
	Matrix A;
	Matrix dA;

	void initializeBiasWithZeros();
	void initializeWeightsRandomly();

	void computeAndStoreBackpropError(Matrix& dZ);
	void computeAndStoreLayerOutput(Matrix& A);
	void updateWeights(Matrix& dZ, float learning_rate);
	void updateBias(Matrix& dZ, float learning_rate);

public:
	LinearLayer(std::string name, Shape W_shape);
	~LinearLayer();

	/// <summary>
	/// Calculates z = W * A + b
	/// </summary>
	/// <param name="A">Input of the layer</param>
	/// <returns></returns>
	Matrix& forward(Matrix& A);
	
	/// <summary>
	/// Computes the errors and updates wieghts and bias.
	/// dA = W' * dZ
	/// dW = 1 / m * dZ * A'
	/// db = 1 / m * Sum(dZ[i])
	/// </summary>
	/// <param name="dZ">Error from the previous layers</param>
	/// <param name="learningRate">The learning rate.</param>
	/// <returns>dA</returns>
	Matrix& backprop(Matrix& dZ, float learning_rate = 0.01);

	int getXDim() const;
	int getYDim() const;

	Matrix getWeightsMatrix() const;
	Matrix getBiasVector() const;
};
