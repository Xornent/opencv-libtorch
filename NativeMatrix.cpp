
#include "pch.h"
#include "NativeMatrix.h"

int NativeMatrix::allocate() {
	cv::Mat mat;
	matrixStore.push_back(mat);
	return matrixStore.size() - 1;
}

int NativeMatrix::allocate(std::string path) {
	matrixStore.push_back(cv::imread(path));
	return matrixStore.size() - 1;
}

int NativeMatrix::allocate(cv::Mat matrix) {
	matrixStore.push_back(matrix);
	return matrixStore.size() - 1;
}

cv::Mat NativeMatrix::get(int index) {
	return matrixStore[index];
}

void NativeMatrix::reinitialize() {
	matrixStore.erase(matrixStore.begin());
	std::vector< cv::Mat > matrixSwap;
	matrixStore.swap(matrixSwap);
}