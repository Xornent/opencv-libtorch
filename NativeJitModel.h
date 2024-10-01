#pragma once
#ifndef H_NATIVEJITMODEL
#define H_NATIVEJITMODEL

#include "pch.h"

class NativeJitModel {

public:

	int allocate(std::string path);
	void save(std::string path, int id);

	void train(int id);
	void eval(int id);
	bool is_training(int id);

	int forward(int tensorId, int modelId, bool autoGrad);

	void reinitialize();

};

#endif