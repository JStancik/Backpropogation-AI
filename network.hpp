#pragma once
#include <ctime>
#include <vector>

struct Neuron{
	Neuron():error(0),value(0){};
	std::vector<float> weights;
	float error;
	float value;
};

struct Filter{
	Filter(){};
	std::vector<std::vector<float>> weights;
	int stride;
	bool fill;
};

struct PoolLayer{
	int size;
	int stride;
	bool isAvg;
};

class AINet{
	public:
		AINet(int inputCount,int hiddenLCount, int hiddenCount, int outputCount);
		std::vector<float> getOutput(std::vector<float> input);
		void               backPropogate(std::vector<float> &inputs,std::vector<float> &output,std::vector<float> &expected,float &learnSpeed);
		std::vector<float> trainNet(std::vector<float> input,std::vector<float> expected, float learnSpeed,int debugLevel = 0);
		void               trainNet(std::string dataFile,int dataCount,int iterations,    float learnSpeed,int debugLevel = 1);
		void               trainNet(std::vector<std::vector<float>> inputs,std::vector<std::vector<float>> expected,int dataCount,int iterations,float learnSpeed,int debugLevel = 1);

	private:
		void clearHiddenLayer();
		std::vector<std::vector<Neuron>> hiddenLayers;
		std::vector<Neuron> outputLayer;
		clock_t debugTime;
		int lastTimeStep;
		int forwardPropTime;
		int BackwardPropTime;
		int miscTime;
};

class CNN{
	public:
		CNN(int inputWidth,int inputHeight,int outputCount);
		void  addConvLayer(int filterCount,int filterSize,int stride,bool fill);
		void  addPoolLayer(int size,int stride,bool isAvg);
		void addDenseLayer(int nCount);
		std::vector<float> getOutput(auto input);
		std::vector<float> trainNet (auto input,std::vector<float> expected, float learnSpeed,int debugLevel = 0);
		void               trainNet (auto inputs,std::vector<std::vector<float>> expected,int dataCount,int iterations,float learnSpeed,int debugLevel = 1);
	private:
		std::vector<std::vector<Filter>> convLayers;
		std::vector<PoolLayer> poolLayers;
		std::vector<std::vector<Neuron>> denseLayers;

		std::vector<uint8_t> layerTypeOrder;
};

inline float tanh_derivative(float x){
	return 1-tanh(x)*tanh(x);
}

inline float LReLU(float x){
	return std::max(x,0.01f*x);
}

inline float LReLU_derivative(float x){
	return x<0?0.01:1;
}