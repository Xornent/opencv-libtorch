
#include "NativeTensor.h"
#include "OpenCV.Native.h"
#include <vector>

NativeTensor::NativeTensor() {

};

int NativeTensor::allocate() {
	size_t size = storedTensors.size();
	storedTensors.push_back(torch::zeros({ 0 }));
	return size;
};

int NativeTensor::allocate_rand(int64_t* dimensions, int length) {
	size_t size = storedTensors.size();
	storedTensors.push_back(torch::rand(
		c10::ArrayRef<int64_t>(dimensions, length)
	));
	return size;
};

int NativeTensor::allocate_ones(int64_t* dimensions, int length) {
	size_t size = storedTensors.size();
	storedTensors.push_back(torch::ones(
		c10::ArrayRef<int64_t>(dimensions, length)
	));
	return size;
};

int NativeTensor::allocate_zeros(int64_t* dimensions, int length) {
	size_t size = storedTensors.size();
	storedTensors.push_back(torch::zeros(
		c10::ArrayRef<int64_t>(dimensions, length)
	));
	return size;
};

int NativeTensor::allocate_from_blobs(int64_t* dimensions, int length, float* values) {
	int64_t elementLength = 1;
	for (int id = 0; id < length; id++)
		elementLength *= dimensions[id];

	auto input = torch::from_blob(values, { elementLength });
	input = input.reshape(c10::ArrayRef<int64_t>(dimensions, length));
	size_t size = storedTensors.size();
	storedTensors.push_back(input);
	return size;
};

int NativeTensor::allocate_from_mat(cv::Mat mat) {
	auto height = mat.rows;
	auto width = mat.cols;
	auto channels = mat.channels();

	auto original = torch::from_blob(mat.data, { height, width, channels }, torch::kFloat);
	original = original.permute({ 2, 0, 1 });
	size_t size = storedTensors.size();
	storedTensors.push_back(original);
	return size;
};

void NativeTensor::free(int index) {
	std::free(&storedTensors[index]);
}

bool NativeTensor::is_cuda(int index) {
	return storedTensors[index].is_cuda();
}

void NativeTensor::cuda(int index) {
	storedTensors[index] = storedTensors[index].cuda();
}

void NativeTensor::cpu(int index) {
	storedTensors[index] = storedTensors[index].cpu();
}

bool NativeTensor::requires_grad(int index) {
	return storedTensors[index].requires_grad();
}

int NativeTensor::backward(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index]);
	return size;
}

int NativeTensor::gt(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] > storedTensors[id2]);
	return size;
}

int NativeTensor::gte(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] >= storedTensors[id2]);
	return size;
}

int NativeTensor::lt(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] < storedTensors[id2]);
	return size;
}

int NativeTensor::lte(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] <= storedTensors[id2]);
	return size;
}

int NativeTensor::add(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] + storedTensors[id2]);
	return size;
}

int NativeTensor::substract(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] - storedTensors[id2]);
	return size;
}

int NativeTensor::multiply(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] * storedTensors[id2]);
	return size;
}

int NativeTensor::divide(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1] / storedTensors[id2]);
	return size;
}

int NativeTensor::pow(int id1, int id2) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[id1].pow( storedTensors[id2] ));
	return size;
}

int NativeTensor::exp(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].exp());
	return size;
}

int NativeTensor::sin(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].sin());
	return size;
}

int NativeTensor::cos(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].cos());
	return size;
}

int NativeTensor::tan(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].tan());
	return size;
}

int NativeTensor::sinh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].sinh());
	return size;
}

int NativeTensor::cosh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].cosh());
	return size;
}

int NativeTensor::tanh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].tanh());
	return size;
}

int NativeTensor::arcsinh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].asinh());
	return size;
}

int NativeTensor::arccosh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].acosh());
	return size;
}

int NativeTensor::arctanh(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].atanh());
	return size;
}

int NativeTensor::arcsin(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].asin());
	return size;
}

int NativeTensor::arccos(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].acos());
	return size;
}

int NativeTensor::arctan(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].atan());
	return size;
}

int NativeTensor::sgn(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].sgn());
	return size;
}

int NativeTensor::abs(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].abs());
	return size;
}

int NativeTensor::relu(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].relu());
	return size;
}

int NativeTensor::ln(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].log());
	return size;
}

int NativeTensor::lg(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].log10());
	return size;
}

int NativeTensor::log(int index, float base) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].log10() / log10(base));
	return size;
}

int NativeTensor::sigmoid(int index) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].sigmoid());
	return size;
}

int NativeTensor::softmax(int index, int dim) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].softmax(dim, c10::kFloat));
	return size;
}

int NativeTensor::log_softmax(int index, int dim) {
	size_t size = storedTensors.size();
	storedTensors.push_back(storedTensors[index].log_softmax(dim, c10::kFloat));
	return size;
}

std::vector<int64_t> NativeTensor::size(int index) {
	auto s = storedTensors[index].sizes();
	std::vector<int64_t> vector = {};
	for (int id = 0; id < s.size(); id++)
		vector.push_back(s[id]);
	return vector;
}

int64_t NativeTensor::length(int index) {
	auto s = storedTensors[index].sizes();
	int64_t length = 1;
	for (int id = 0; id < s.size(); id++)
		length *= s[id];
	return length;
}

float* NativeTensor::to_float_array(int index) {
	auto s = storedTensors[index].sizes();
	long length = 1;
	for (int id = 0; id < s.size(); id++)
		length *= s[id];

	auto line = storedTensors[index].reshape({ length });
	if (s.size() == 0) length = 0;
	float* nativeArray = (float*)calloc(sizeof(float), length);
	for (int id = 0; id < length; id++)
		nativeArray[id] = line[id].item().toFloat();

	return nativeArray;
}

// input 1x512x512.
std::vector<int> NativeTensor::util_generate_clip_and_process
(int targetSize, int paddingSize, int batch, cv::Mat mat, int modelId) {

	torch::NoGradGuard nograd;

	int totaly = ((mat.rows - 2 * paddingSize) / (targetSize - 2 * paddingSize));
	int totalx = ((mat.cols - 2 * paddingSize) / (targetSize - 2 * paddingSize));
	int width = mat.cols;
	int height = mat.rows;
	auto model = jitModules[modelId];
	std::cout << "Begin Clipping Images of: [Width = " << width << 
		", Height = " << height <<
		", Total X Clips = " << totalx << 
		", Total Y Clips = " << totaly << "]" << std::endl;

	std::vector<int> ids;
	std::vector< torch::Tensor > stored;
	for (int y = 0; y < totaly; y++) {
		for (int x = 0; x < totalx; x++) {
			int startx = x * (targetSize - 2 * paddingSize);
			int starty = y * (targetSize - 2 * paddingSize);
			std::cout << "Currently Clipping [X =" << startx << ", Y =" << starty << "]" << std::endl;

			cv::Rect clipRect(startx, starty, targetSize, targetSize);
			cv::Mat interest = mat(clipRect);

			float* inputData = (float*)malloc(sizeof(float) * targetSize * targetSize);
			int _index = 0;
			for (int _y = 0; _y < targetSize; _y++) {
				uchar* rowPointer = interest.ptr<uchar>(_y);
				for (int _x = 0; _x < targetSize; _x++) {
					inputData[_index] = rowPointer[_x] * 1.0f;
					_index++;
				}
			}

			auto clip = torch::from_blob(inputData, { targetSize * targetSize }, torch::kFloat);
			clip = clip.reshape({1, 1, targetSize, targetSize});

			// std::cout << clip.sizes() << std::endl;

			clip = (clip - torch::mean(clip)) / torch::std(clip);
			// std::cout << clip << std::endl;
			stored.push_back(clip);
		}
	}

	std::cout << "Begin Batching" << std::endl;

	for (int id = 0; id < stored.size(); id += batch) {
		torch::Tensor nativeArray[15];
		for (int bid = id; bid < MIN(id + batch, stored.size()); bid++) {
			nativeArray[bid - id] = stored[bid];
		}
		
		auto inp = torch::cat(c10::ArrayRef<torch::Tensor>(nativeArray, MIN(batch, stored.size() - id)), 0);
		std::cout << "Batch from " << id << " to " << MIN(id + batch, stored.size()) - 1 << ": Tensor Size " << inp.sizes() << std::endl;
		
		model.eval();
		auto output = model.forward({ inp.to(torch::kCPU) }).toTensor();

		int size = storedTensors.size();
		storedTensors.push_back(output);
		ids.push_back(size);

		// std::cout << id << "/" << stored.size() << std::endl;
	}

	return ids;
}

void NativeTensor::reinitialize() {
	storedTensors.erase(storedTensors.begin());
	std::vector <torch::Tensor> swapStore;
	storedTensors.swap(swapStore);
}

NativeTensor::~NativeTensor() {

};

void NativeTensor::print(int index) {
	std::cout << storedTensors[index] << std::endl;
}