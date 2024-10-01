#pragma once
#ifndef H_JITMODEL
#define H_JITMODEL

#include "NativeJitModel.h"
#include "Tensor.h"

namespace Torch {

	public ref class JitModel {

	public:

		JitModel(System::String^ path);

		void Save(System::String^ path);
		Tensor^ Forward(Tensor^ input, bool autoGrad);
		System::Boolean IsTraining();
		void Train();
		void Eval();

		Int32 index;

		static void Reinitialize();

	internal:

		JitModel(Int32 id);

	};

}

#endif