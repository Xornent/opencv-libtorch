#pragma once
#ifndef H_MATIVEMATRIX
#define H_MATIVEMATRIX

#include "pch.h"

static std::vector< cv::Mat > matrixStore;

class NativeMatrix {

public:

	int allocate(std::string path);
	int allocate(cv::Mat matrix);
	int allocate();
	cv::Mat get(int index);

	void reinitialize();

};

#endif