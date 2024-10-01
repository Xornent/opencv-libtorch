
#include "OpenCV.Native.h"
#include "NativeJitModel.h"

int NativeJitModel::allocate(std::string path) {
	size_t size = jitModules.size();
	auto model = torch::jit::load(path);
	jitModules.push_back(model);
	return size;
}

void NativeJitModel::save(std::string path, int id) {
	jitModules[id].save(path);
}

int NativeJitModel::forward(int tensorId, int modelId, bool autoGrad) {
	if (!autoGrad) {
		torch::NoGradGuard nograd;
		storedTensors[tensorId].requires_grad_(false);
	}

	torch::Tensor result = jitModules[modelId].forward({ storedTensors[tensorId] }).toTensor();
	size_t size = storedTensors.size();
	storedTensors.push_back(result);
	return size;
}

void NativeJitModel::train(int id) {
	jitModules[id].train();
}

void NativeJitModel::eval(int id) {
	jitModules[id].eval();
}

bool NativeJitModel::is_training(int id) {
	return jitModules[id].is_training();
}

void NativeJitModel::reinitialize() {
	jitModules.erase(jitModules.begin());
	std::vector <torch::jit::Module> swapStore;
	jitModules.swap(swapStore);
}
