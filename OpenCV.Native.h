#ifndef H_OCVNATIVE
#define H_OCVNATIVE

#include "pch.h"
#include <torch/torch.h>
#include <torch/script.h>

extern std::vector< torch::Tensor > storedTensors;
extern std::vector< torch::jit::Module > jitModules;

#endif