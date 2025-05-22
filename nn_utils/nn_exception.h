#pragma once

#include <exception>
#include <iostream>

class NNException : std::exception {
private:

	/// <summary>
	/// The error message.
	/// </summary>
	const char* exception_message;

public:

	/// <summary>
	/// Initializes a new instance of the <see cref="NNException"/> class with the given error message.
	/// </summary>
	/// <param name="exception_message">The error message.</param>
	NNException(const char* exception_message) :
		exception_message(exception_message)
	{ }

	/// <summary>
	/// Returns the error message.
	/// </summary>
	/// <returns>Error message</returns>
	virtual const char* what() const throw()
	{
		return exception_message;
	}

	/// <summary>
	/// Checks for CUDA errors and throws exception if any occured.
	/// </summary>
	/// <param name="exception_message">The error message to throw.</param>
	static void throwIfDeviceErrorsOccurred(const char* exception_message) {
		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			std::cerr << error << ": " << exception_message;
			throw NNException(exception_message);
		}
	}
};
