
#include "pch.h"
#include "NativeJitModel.h"
#include "JitModel.h"
#include "Tensor.h"

using namespace System;
using namespace Runtime::InteropServices;

static NativeJitModel nativeJitModel;

void MarshalString(String^ s, std::string& outputstring)
{
	const char* kPtoC = (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	outputstring = kPtoC;
	Marshal::FreeHGlobal(IntPtr((void*)kPtoC));
}

Torch::JitModel::JitModel(System::String^ path) {
	std::string cstr;
	MarshalString(path, cstr);
	int id = nativeJitModel.allocate(cstr);
	this->index = id;
}

Torch::JitModel::JitModel(System::Int32 id) {
	this->index = id;
}

void Torch::JitModel::Train() {
	nativeJitModel.train(this->index);
}

void Torch::JitModel::Eval() {
	nativeJitModel.eval(this->index);
}

System::Boolean Torch::JitModel::IsTraining() {
	return nativeJitModel.is_training(this->index);
}

void Torch::JitModel::Save(System::String^ path) {
	std::string cstr;
	MarshalString(path, cstr);
	nativeJitModel.save(cstr, this->index);
}

Torch::Tensor^ Torch::JitModel::Forward(Torch::Tensor^ input, bool autoGrad) {
	int index = nativeJitModel.forward(input->index, this->index, autoGrad);
	return gcnew Torch::Tensor(index);
}

void Torch::JitModel::Reinitialize() {
	nativeJitModel.reinitialize();
}