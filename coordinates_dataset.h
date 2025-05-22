#pragma once

#include "nn_utils/matrix.h"

#include <vector>

class CoordinatesDataset {
private:
	size_t batch_size;
	size_t number_of_batches;

	std::vector<Matrix> batches;
	std::vector<Matrix> targets;

public:

	/// <summary>
	/// Initializes a new instance of the <see cref="TestDataset"/> class.
	/// The dataset creates random points in a two dimsnional space. If both coordinates are negative or positive
	/// the point is labeled as 1, 0 otherwise
	/// </summary>
	/// <param name="batch_size">Size of the batch.</param>
	/// <param name="number_of_batches">The number of batches.</param>
	CoordinatesDataset(size_t batch_size, size_t number_of_batches);

	int getNumOfBatches();
	std::vector<Matrix>& getBatches();
	std::vector<Matrix>& getTargets();

};
