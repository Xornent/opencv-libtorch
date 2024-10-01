#pragma once
#ifndef H_NATIVETENSOR
#define H_NATIVETENSOR

#include "pch.h"

class NativeTensor {

public:

	int allocate();
	int allocate_rand(int64_t* dimensions, int length);
	int allocate_zeros(int64_t* dimensions, int length);
	int allocate_ones(int64_t* dimensions, int length);
	int allocate_from_blobs(int64_t* dimensions, int length, float* values);
	int allocate_from_mat(cv::Mat mat);

	void free(int index);

	bool is_cuda(int index);
	void cuda(int index);
	void cpu(int index);

	bool requires_grad(int index);

	int backward(int index);

	int add(int id1, int id2);
	int substract(int id1, int id2);
	int multiply(int id1, int id2);
	int divide(int id1, int id2);
	int gt(int id1, int id2);
	int lt(int id1, int id2);
	int gte(int id1, int id2);
	int lte(int id1, int id2);
	int pow(int id1, int id2);
	
	int exp(int id1);
	int sin(int index);
	int cos(int index);
	int tan(int index);
	int sinh(int index);
	int cosh(int index);
	int tanh(int index);
	int arcsin(int index);
	int arccos(int index);
	int arctan(int index);
	int arcsinh(int index);
	int arccosh(int index);
	int arctanh(int index);
	int abs(int index);
	int sgn(int index);
	int relu(int index);
	int softmax(int index, int dim);
	int ln(int index);
	int lg(int index);
	int log(int index, float base);
	int log_softmax(int index, int dim);
	int sigmoid(int index);

	std::vector<int64_t> size(int index);

	NativeTensor();
	~NativeTensor();

	void print(int id);
	float* to_float_array(int id);
	int64_t length(int index);

	void reinitialize();

	std::vector<int> util_generate_clip_and_process(int targetSize, int paddingSize, int batch, cv::Mat mat, int modelid);

private:

};

#endif