#pragma once
#ifndef H_MATRIX
#define H_MATRIX

#include "pch.h"
#include "Tensor.h"

namespace OpenCV {

	public enum class PixelDepth {
		SignedByte    = CV_8S,
		UnsignedByte  = CV_8U,
		SignedShort   = CV_16S,
		UnsignedShort = CV_16U,
		SignedInt     = CV_32S,
		Float         = CV_32F,
		Double        = CV_64F
	};

	public enum class ColorConversion {
		RGB_GrayScale = cv::COLOR_RGB2GRAY
	};

	public enum class ThresholdMethod {
		Binary          = cv::THRESH_BINARY,
		BinaryInverted  = cv::THRESH_BINARY_INV,
		Truncate        = cv::THRESH_TRUNC,
		Triangle        = cv::THRESH_TRIANGLE,
		ToZero          = cv::THRESH_TOZERO,
		ToZeroInverted  = cv::THRESH_TOZERO_INV,
		OTSU            = cv::THRESH_OTSU
	};

	public enum class AdaptiveThresholdMethod {
		Mean            = cv::ADAPTIVE_THRESH_MEAN_C,
		Gaussian        = cv::ADAPTIVE_THRESH_GAUSSIAN_C
	};

	public enum class ContourMode {
		External        = cv::RETR_EXTERNAL,
		FloodFill       = cv::RETR_FLOODFILL,
		CComp           = cv::RETR_CCOMP,
		List            = cv::RETR_LIST
	};

	public enum class ContourMethod {
		Simple          = cv::CHAIN_APPROX_SIMPLE,
		None            = cv::CHAIN_APPROX_NONE,
		TehChinKCos     = cv::CHAIN_APPROX_TC89_KCOS,
		TehChinL1       = cv::CHAIN_APPROX_TC89_L1
	};

	public ref class Point2D {

	public:

		Point2D(int x, int y);
		int X;
		int Y;

	};

	public ref class ColorRGB {

	public:

		ColorRGB(int r, int g, int b);
		int R;
		int G;
		int B;

	};

	public ref class Contour {
	
	public:

		System::Double Area;
		OpenCV::Point2D^ Central;
		System::Collections::Generic::List< Point2D^ >^ DataPoints;

		Contour(std::vector< cv::Point > points);

		static Contour^ Rectangle(int left, int top, int width, int height);

	};

	public ref class Histogram {

	public:

		System::Collections::Generic::List< double >^ Data;
		double Minimum;
		double Maximum;
		double Sum;

		Histogram();

	};

	public ref class ContourCollection {

	public:

		System::Collections::Generic::List<Contour^>^ DataPoints;
		ContourCollection(std::vector< std::vector< cv::Point > > points);

	};

	public ref class Matrix {

	public:
		
		Matrix(System::String^ path);
		Matrix(int width, int height, PixelDepth depth, int channel);
		
		System::Byte* Data;
		System::Byte* RowData(int row);

		System::Int32 Channels;
		System::Int32 Width;
		System::Int32 Height;
		PixelDepth Depth;

		Matrix^ ConvertColor(ColorConversion conversion);
		Matrix^ ExtractChannel(int channel);

		Matrix^ Threshold(double threshold, double maxValue, ThresholdMethod method);
		Matrix^ AdaptiveThreshold(double maxValue, AdaptiveThresholdMethod adaptive, ThresholdMethod method, int blockSize, double offset);

		ContourCollection^ FindContours(ContourMode mode, ContourMethod method);

		Histogram^ CalculateHistogram(Matrix^ mask, int channel, int offsetX, int offsetY);
		OpenCV::Point2D^ OpenCV::Matrix::FindAdaptiveOffset(OpenCV::Matrix^ mask, int channel, int adaptiveRegion);
		OpenCV::Matrix^ PaddingReflect(int left, int top, int right, int bottom);

		void Fill(ColorRGB^ rgb);
		void FillContour(ColorRGB^ rgb, Contour^ contour);

		void Save(System::String^ path);
		Torch::Tensor^ ToTensor();

		static void Reinitialize();

		List<Torch::Tensor^>^ SPFND_UTIL_GEN_CL_PROC(int targerSize, int paddingSize, int batch, int model);

	private:

		int Index;
		Matrix(cv::Mat& matrix);
		~Matrix();

	};

}

#endif