#pragma once
#ifndef H_OCVINTEROP
#define H_OCVINTEROP

using namespace System;

namespace Xornent {

	public ref class Version
	{

	public:

		int Major;
		int Minor;
		int Build;

		Version(int major, int minor, int build);

	};

	namespace OpenCV {

		public ref class Version {

		public:

			Xornent::Version^ OpenCV = gcnew Xornent::Version(4, 5, 4);
			Xornent::Version^ Wrapper = gcnew Xornent::Version(0, 1, 0);
		};

	}

	namespace Torch {

		public ref class Version {

		public:

			Xornent::Version^ Torch = gcnew Xornent::Version(1, 7, 0);
			Xornent::Version^ Wrapper = gcnew Xornent::Version(0, 3, 0);
		};

	}
}

#endif