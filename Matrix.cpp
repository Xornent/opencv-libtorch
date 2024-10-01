
#include "pch.h"
#include "Matrix.h"
#include "NativeMatrix.h"

using namespace System;
using namespace Runtime::InteropServices;
using namespace System::Collections::Generic;

static NativeMatrix nativeMatrix;

static void marshalString(System::String^ s, std::string& outputstring)
{
	const char* kPtoC = (const char*)(Marshal::StringToHGlobalAnsi(s)).ToPointer();
	outputstring = kPtoC;
	Marshal::FreeHGlobal(IntPtr((void*)kPtoC));
}

cv::Mat translateImage(cv::Mat& matSrc, int xOffset, int yOffset, bool bScale)
{
	int nRows = matSrc.rows;
	int nCols = matSrc.cols;
	int nRowsRet = 0;
	int nColsRet = 0;
	cv::Rect rectSrc;
	cv::Rect rectRet;

	if (bScale) {
		nRowsRet = nRows + abs(yOffset);
		nColsRet = nCols + abs(xOffset);
		rectSrc.x = 0;
		rectSrc.y = 0;
		rectSrc.width = nCols;
		rectSrc.height = nRows;
	}
	else {
		nRowsRet = matSrc.rows;
		nColsRet = matSrc.cols;
		if (xOffset >= 0) rectSrc.x = 0;
		else rectSrc.x = abs(xOffset);

		if (yOffset >= 0) rectSrc.y = 0;
		else rectSrc.y = abs(yOffset);

		rectSrc.width = nCols - abs(xOffset);
		rectSrc.height = nRows - abs(yOffset);
	}

	if (xOffset >= 0) rectRet.x = xOffset;
	else rectRet.x = 0;

	if (yOffset >= 0) rectRet.y = yOffset;
	else rectRet.y = 0;

	rectRet.width = rectSrc.width;
	rectRet.height = rectSrc.height;

	cv::Mat matRet(nRowsRet, nColsRet, matSrc.type(), cv::Scalar(0));
	matSrc(rectSrc).copyTo(matRet(rectRet));
	return matRet;
}

OpenCV::Matrix::Matrix(cv::Mat& matrix) {
	int id = nativeMatrix.allocate(matrix);
	this->Index = id;
	cv::Mat* _matrixPtr = &nativeMatrix.get(id);
	this->Channels = _matrixPtr->channels();
	this->Depth = (PixelDepth)(_matrixPtr->depth());
	this->Width = _matrixPtr->cols;
	this->Height = _matrixPtr->rows;
}

OpenCV::Matrix::Matrix(String^ path) {
	std::string cstring;
	marshalString(path, cstring);
	std::cout << cstring << std::endl;
	int id = nativeMatrix.allocate(cstring);
	cv::Mat m = nativeMatrix.get(id);
	this->Index = id;
	this->Channels = m.channels();
	this->Depth = (PixelDepth)(m.depth());
	this->Width = m.cols;
	this->Height = m.rows;
}

OpenCV::Matrix::Matrix(int width, int height, PixelDepth depth, int channel) {
	cv::Mat matrix = cv::Mat(height, width, CV_MAKETYPE(((int)depth), channel));
	int id = nativeMatrix.allocate(matrix);
	this->Index = id;
	cv::Mat* _matrixPtr = &nativeMatrix.get(id);
	this->Data = _matrixPtr->data;
	this->Channels = _matrixPtr->channels();
	this->Depth = (PixelDepth)(_matrixPtr->depth());
	this->Width = _matrixPtr->cols;
	this->Height = _matrixPtr->rows;
}

Byte* OpenCV::Matrix::RowData(int row) {
	return nativeMatrix.get(this->Index).ptr<uchar>(row);
}

OpenCV::Matrix^ OpenCV::Matrix::ConvertColor(ColorConversion conversion) {
	cv::Mat result;
	cv::cvtColor(nativeMatrix.get(this->Index), result, (int)conversion);
	return gcnew OpenCV::Matrix(result);
}

OpenCV::Matrix^ OpenCV::Matrix::ExtractChannel(int channel) {
	cv::Mat result;
	cv::extractChannel(nativeMatrix.get(this->Index), result, channel);
	return gcnew OpenCV::Matrix(result);
}

OpenCV::Matrix^ OpenCV::Matrix::Threshold(double threshold, double maxValue, ThresholdMethod method) {
	cv::Mat result;
	cv::threshold(nativeMatrix.get(this->Index), result, threshold, maxValue, (int)method);
	return gcnew OpenCV::Matrix(result);
}

OpenCV::Matrix^ OpenCV::Matrix::AdaptiveThreshold(double maxValue, AdaptiveThresholdMethod adaptive, ThresholdMethod method, int blockSize, double offset) {
	cv::Mat result;
	cv::adaptiveThreshold(nativeMatrix.get(this->Index), result, maxValue, (int)adaptive, (int)method, blockSize, offset);
	return gcnew OpenCV::Matrix(result);
}

OpenCV::ContourCollection::ContourCollection( std::vector<std::vector<cv::Point>> points ) {
	this->DataPoints = gcnew List<Contour^>();
	for (int idContours = 0; idContours < points.size(); idContours++) {
		this->DataPoints->Add(gcnew OpenCV::Contour(points[idContours]));
	}
}

OpenCV::Contour::Contour(std::vector<cv::Point> points) {
	this->DataPoints = gcnew List<Point2D^>();
	for (int idPoints = 0; idPoints < points.size(); idPoints++) {
		this->DataPoints->Add(gcnew OpenCV::Point2D(points[idPoints].x, points[idPoints].y));
	}
	cv::Moments moment;
	moment = cv::moments(points);
	this->Central = gcnew OpenCV::Point2D(moment.m10 / moment.m00, moment.m01 / moment.m00);
	this->Area = cv::contourArea(points);
}

void OpenCV::Matrix::Save(System::String^ path) {
	std::string cstring;
	marshalString(path, cstring);
	cv::imwrite(cstring, nativeMatrix.get(this->Index));
}

OpenCV::Point2D::Point2D(int x, int y) {
	this->X = x;
	this->Y = y;
}

OpenCV::ContourCollection^ OpenCV::Matrix::FindContours(ContourMode mode, ContourMethod method) {
	std::vector <std::vector <cv::Point>> contours;
	cv::findContours(nativeMatrix.get(this->Index), contours, (int)mode, (int)method);
	return gcnew OpenCV::ContourCollection(contours);
}

OpenCV::ColorRGB::ColorRGB(int r, int g, int b) {
	this->R = r;
	this->G = g;
	this->B = b;
}

void OpenCV::Matrix::Fill(ColorRGB^ rgb) {
	auto matrix = nativeMatrix.get(this->Index);
	cv::rectangle(matrix, cv::Point(0, 0), cv::Point(matrix.cols, matrix.rows),
		CV_RGB(rgb->R, rgb->G, rgb->B), -1, 4);
}

void OpenCV::Matrix::FillContour(ColorRGB^ rgb, Contour^ contour) {
	auto matrix = nativeMatrix.get(this->Index);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Point> con;
	for (int i = 0; i < contour->DataPoints->Count; i++) {
		con.push_back(cv::Point(contour->DataPoints[i]->X, contour->DataPoints[i]->Y));
	}

	contours.push_back(con);
	cv::drawContours(matrix, contours, 0, CV_RGB(rgb->R, rgb->G, rgb->B), -1, 4);
}

OpenCV::Histogram::Histogram() {}

OpenCV::Histogram^ OpenCV::Matrix::CalculateHistogram(OpenCV::Matrix^ mask, int channel, int offsetX, int offsetY) {
	auto matrix = nativeMatrix.get(this->Index);
	int channelInp[1] = { channel };
	const int histSize[1] = { 256 };
	float range1d[2] = { 0.0, 255.0 };
	const float* range[1] = { range1d };
	cv::MatND histogram;

	if (mask == nullptr) {
		cv::calcHist(&matrix, 1, channelInp, cv::Mat(), histogram, 1, histSize, range);
	}
	else {
		cv::Mat currentMask = nativeMatrix.get(mask->Index);
		cv::Mat translatedMask = translateImage(currentMask, offsetX, offsetY, false);
		cv::calcHist(&matrix, 1, channelInp, translatedMask, histogram, 1, histSize, range);
		translatedMask.release();
	}

	OpenCV::Histogram^ hist = gcnew OpenCV::Histogram();
	int minLoc, maxLoc;
	double min, max;
	cv::minMaxLoc(histogram, &min, &max);

	hist->Data = gcnew List<double>();
	hist->Sum = 0;
	for (int i = 0; i < histSize[0]; i++) {
		hist->Data->Add(histogram.at<float>(i));
		hist->Sum += i * histogram.at<float>(i);
	}

	hist->Maximum = max;
	hist->Minimum = min;

	return hist;
}

OpenCV::Point2D^ OpenCV::Matrix::FindAdaptiveOffset(OpenCV::Matrix^ mask, int channel, int adaptiveRegion) {
	auto matrix = nativeMatrix.get(this->Index);
	int channelInp[1] = { channel };
	const int histSize[1] = { 256 };
	float range1d[2] = { 0.0, 255.0 };
	const float* range[1] = { range1d };
	int maximumSum = 0;
	int maxX = 0, maxY = 0;

	for(int offsetX = -adaptiveRegion; offsetX <= adaptiveRegion; offsetX++)
		for (int offsetY = -adaptiveRegion; offsetY <= adaptiveRegion; offsetY++) {
			cv::MatND histogram;
			if (mask == nullptr) {
				cv::calcHist(&matrix, 1, channelInp, cv::Mat(), histogram, 1, histSize, range);
			}
			else {
				cv::Mat currentMask = nativeMatrix.get(mask->Index);
				cv::Mat translatedMask = translateImage(currentMask, offsetX, offsetY, false);
				cv::calcHist(&matrix, 1, channelInp, translatedMask, histogram, 1, histSize, range);
			}

			int sum = 0;
			for (int i = 0; i < histSize[0]; i++) 
				sum += i * histogram.at<float>(i);
			
			if (sum > maximumSum) {
				maximumSum = sum;
				maxX = offsetX;
				maxY = offsetY;
			}
		}

	return gcnew OpenCV::Point2D(maxX, maxY);
}

OpenCV::Contour^ OpenCV::Contour::Rectangle(int left, int top, int width, int height) {
	std::vector< cv::Point > points;
	points.push_back(cv::Point(left, top));
	points.push_back(cv::Point(left + width, top));
	points.push_back(cv::Point(left + width, top + height));
	points.push_back(cv::Point(left, top + height));
	return gcnew OpenCV::Contour(points);
}

Torch::Tensor^ OpenCV::Matrix::ToTensor() {
	return gcnew Torch::Tensor(nativeMatrix.get(Index));
}

OpenCV::Matrix^ OpenCV::Matrix::PaddingReflect(int left, int top, int right, int bottom) {
	cv::Mat result;
	cv::copyMakeBorder(nativeMatrix.get(this->Index), result, top, bottom, left, right, cv::BORDER_REFLECT);
	return gcnew OpenCV::Matrix(result);
}

List<Torch::Tensor^>^ OpenCV::Matrix::SPFND_UTIL_GEN_CL_PROC(int targerSize, int paddingSize, int batch, int model) {
	auto ids = nativeTensor.util_generate_clip_and_process(targerSize, paddingSize, batch, nativeMatrix.get(this->Index), model);
	List<Torch::Tensor^>^ list = gcnew List<Torch::Tensor^>();
	for (int i = 0; i < ids.size(); i++)
		list->Add(gcnew Torch::Tensor(ids[i]));
	return list;
}

void OpenCV::Matrix::Reinitialize() {
	nativeMatrix.reinitialize();
}

OpenCV::Matrix::~Matrix() {
	nativeMatrix.get(this->Index).release();
}