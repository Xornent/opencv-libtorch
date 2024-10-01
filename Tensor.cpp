
#include "pch.h"
#include "NativeTensor.h"
#include "Tensor.h"

Torch::Tensor::Tensor(Int32 id) {
	this->index = id;
	auto list = gcnew System::Collections::Generic::List<long>();
	auto sizes = nativeTensor.size(id);
	for (int i = 0; i < sizes.size(); i++)
		list->Add(sizes[i]);
	this->Size = list;
};

Torch::Tensor::Tensor(List<long>^ dimensions, InitializingPattern pattern) {
	int size = dimensions->Count;
	int64_t* nativeArray = (int64_t*)calloc(sizeof(int64_t), size);

	for (int id = 0; id < size; id++)
		nativeArray[id] = dimensions[id];

	switch (pattern)
	{
	case InitializingPattern::Ones:
		this->index = nativeTensor.allocate_ones(nativeArray, size);
		break;
	case InitializingPattern::Zeros:
		this->index = nativeTensor.allocate_zeros(nativeArray, size);
		break;
	case InitializingPattern::Random:
		this->index = nativeTensor.allocate_rand(nativeArray, size);
		break;

	default:
		this->index = nativeTensor.allocate_ones(nativeArray, size);
		break;
	}
	
	this->Size = dimensions;

	free(nativeArray);
	nativeArray = NULL;
}

Torch::Tensor::Tensor(List<long>^ dimensions, List<float>^ data) {
	int size = dimensions->Count;
	int64_t* nativeArray = (int64_t*)calloc(sizeof(int64_t), size);
	float* datas = (float*)calloc(sizeof(float), data->Count);

	for (int id = 0; id < size; id++)
		nativeArray[id] = dimensions[id];
	for (int id = 0; id < data->Count; id++)
		datas[id] = data[id];

	this->index = nativeTensor.allocate_from_blobs(nativeArray, size, datas);
	this->Size = dimensions;

	free(nativeArray);
	nativeArray = NULL;
}

Torch::Tensor::Tensor(cv::Mat& mat) {
	this->index = nativeTensor.allocate_from_mat(mat);

	this->Size = gcnew System::Collections::Generic::List<long>();
	this->Size->Add((long)(mat.channels()));
	this->Size->Add((long)(mat.rows));
	this->Size->Add((long)(mat.cols));
}

Torch::Tensor^ Torch::Tensor::operator+(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.add(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator-(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.substract(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator*(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.multiply(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator/(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.divide(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator<(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.lt(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator<=(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.lte(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator>(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.gt(a->index, b->index));
}

Torch::Tensor^ Torch::Tensor::operator>=(Tensor^ a, Tensor^ b) {
	return gcnew Torch::Tensor(nativeTensor.gte(a->index, b->index));
}

void Torch::Tensor::Print() {
	nativeTensor.print(this->index);
};

List<float>^ Torch::Tensor::GetValueList() {
	auto arr = nativeTensor.to_float_array(this->index);
	auto listed = gcnew List<float>();
	for (int i = 0; i < nativeTensor.length(this->index); i++) {
		listed->Add(arr[i]);
	}
	return listed;
}

int Torch::Tensor::GetIndex(List<int>^ indices) {
	List<long long>^ capacity = gcnew List<long long>();
	long long temp = 1;
	for (int id = this->Size->Count - 1; id >= 0; id--) {
		capacity->Insert(0, temp);
		temp *= this->Size[id];
	}

	int index = 0;
	if (indices->Count == capacity->Count) {
		for (int id = 0; id < indices->Count; id++)
			index = MIN(INT_MAX, index + indices[id] * capacity[id]);
		return index;
	}
	else return -1;
}

long long Torch::Tensor::GetIndex(List<long long>^ indices) {
	List<long long>^ capacity = gcnew List<long long>();
	long long temp = 1;
	for (int id = this->Size->Count - 1; id >= 0; id--) {
		capacity->Insert(0, temp);
		temp *= this->Size[id];
	}

	long long index = 0;
	if (indices->Count == capacity->Count) {
		for (int id = 0; id < indices->Count; id++)
			index += indices[id] * capacity[id];
		return index;
	}
	else return -1;
}

void Torch::Tensor::Reinitialize() {
	nativeTensor.reinitialize();
}