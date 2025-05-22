#pragma once
#include "matrix.h"

class BCECost {
public:
	/// <summary>
	/// Binary Cross Entropy loss
	/// </summary>
	/// <param name="predictions">Predictions of the neural network</param>
	/// <param name="target">Desired output of the network</param>
	/// <returns></returns>
	float cost(Matrix predictions, Matrix target);
	
	/// <summary>
	/// Derivative of the BCE accordingly the target
	/// </summary>
	/// <param name="predictions">Predictions of the neural network</param>
	/// <param name="target">Desired output of the network</param>
	/// <param name="dY">Derivative</param>
	/// <returns></returns>
	Matrix dCost(Matrix predictions, Matrix target, Matrix dY);
};
