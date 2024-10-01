#pragma once
#ifndef H_TENSOR
#define H_TENSOR

#include "NativeTensor.h"

using namespace System;
using namespace System::Collections::Generic;

static NativeTensor nativeTensor;

namespace Torch {

	public enum class InitializingPattern {
		Zeros,
		Ones,
		Random
	};

	public ref class Tensor {

	public:

		Tensor(List<long>^ dimensions, InitializingPattern pattern);
		Tensor(List<long>^ dimensions, List<float>^ data);

		static Tensor^ operator+(Tensor^ a, Tensor^ b);
		static Tensor^ operator-(Tensor^ a, Tensor^ b);
		static Tensor^ operator*(Tensor^ a, Tensor^ b);
		static Tensor^ operator/(Tensor^ a, Tensor^ b);
		static Tensor^ operator>(Tensor^ a, Tensor^ b);
		static Tensor^ operator>=(Tensor^ a, Tensor^ b);
		static Tensor^ operator<(Tensor^ a, Tensor^ b);
		static Tensor^ operator<=(Tensor^ a, Tensor^ b);

		void Print();
		List<float>^ GetValueList();

		int GetIndex(List<int>^ indices);
		long long GetIndex(List<long long>^ indices);

		static void Reinitialize();

		List<long>^ Size;

	internal:

		Tensor(Int32 id);

		Tensor(cv::Mat& matrix);
		Int32 index;

	};

}

#endif