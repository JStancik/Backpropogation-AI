#pragma once
#include <random>
#include "network.hpp"
#include "dataUtils.hpp"
#include <array>

AINet::AINet(int inputCount,int hiddenLCount, int hiddenCount, int outputCount){
	for(int i=0;i<hiddenLCount;++i){
		hiddenLayers.push_back(std::vector<Neuron>{});
		for(int j=0;j<hiddenCount;++j){
			hiddenLayers[i].push_back(Neuron());
			int neuronCount = i==0?inputCount:hiddenCount;
			for(int k=0;k<neuronCount;++k){
				hiddenLayers[i][j].weights.push_back((float)rand()/RAND_MAX);
			}
			hiddenLayers[i][j].weights.push_back(0);
		}
	}
	for(int i=0;i<outputCount;++i){
		outputLayer.push_back(Neuron());
		for(int j=0;j<hiddenCount;++j){
			outputLayer[i].weights.push_back((float)rand()/RAND_MAX);
		}
		outputLayer[i].weights.push_back((float)rand()/RAND_MAX);
	}
}

std::vector<float> AINet::getOutput(std::vector<float> inputs){
	for(int i=0;i<hiddenLayers.size();++i){
		std::vector<float> input;
		if(i!=0){
			for(int j=0;j<hiddenLayers[i-1].size();++j){
				input.push_back(hiddenLayers[i-1][j].value);
			}
		}
		else{
			for(int j=0;j<inputs.size();++j){
				input.push_back(tanh(inputs[j]));
			}
		}

		for(int j=0;j<hiddenLayers[i].size();++j){
			for(int k=0;k<input.size();++k){
				hiddenLayers[i][j].value += input[k]*hiddenLayers[i][j].weights[k];
			}
			hiddenLayers[i][j].value += hiddenLayers[i][j].weights.back();
			hiddenLayers[i][j].value = tanh(hiddenLayers[i][j].value);
		}
	}
	std::vector<float> output;

	for(int i=0;i<outputLayer.size();++i){
		for(int j=0;j<hiddenLayers.back().size();++j){
			outputLayer[i].value += hiddenLayers.back()[j].value*outputLayer[i].weights[j];
		}
		outputLayer[i].value += outputLayer[i].weights.back();
		outputLayer[i].value = tanh(outputLayer[i].value);
		output.push_back(outputLayer[i].value);
	}
	return output;
}

void AINet::clearHiddenLayer(){
	for(int i=0;i<hiddenLayers.size();++i){
		for(int j=0;j<hiddenLayers[0].size();++j){
			hiddenLayers[i][j].error=0;
		}
	}
}

std::vector<float> AINet::trainNet(std::vector<float> inputs,std::vector<float> expected,float learnSpeed,int debug){
	std::vector<float> output = getOutput(inputs);

	float max = -10;
	int outputIndex;
	int expectedIndex;
	for(int j=0;j<10;++j){
		if(output[j]>max){
			max = output[j];
			outputIndex = j;
		}
		if(expected[j]==1){
			expectedIndex=j;
		}
	}

	if(debug == 2) std::cout<<"output errors:\n";
	for(int i=0;i<outputLayer.size();++i){
		outputLayer[i].error = (outputLayer[i].value-expected[i])*tanh_derivative(outputLayer[i].value);
		if(debug == 2) std::cout<<"OE"<<i<<": "<<outputLayer[i].error<<"\n";
	}
	clearHiddenLayer();
	if(debug == 2) std::cout<<"____________________\nhidden errors:\n";
	for(int j=hiddenLayers.size()-1;j>=0;--j){
		if(debug == 2) std::cout<<"L"<<j<<"\n";
		std::vector<Neuron>* forwardLayer = j==hiddenLayers.size()-1?&outputLayer:&hiddenLayers[j+1];
		for(int k=0;k<hiddenLayers[j].size();++k){
			
			for(int i=0;i<(*forwardLayer).size();++i){
				hiddenLayers[j][k].error += (*forwardLayer)[i].error*(*forwardLayer)[i].weights[k];
			}
		}
		for(int k=0;k<hiddenLayers[j].size();++k){
				hiddenLayers[j][k].error *= tanh_derivative(hiddenLayers[j][k].value);
				if(debug == 2) std::cout<<"  N"<<k<<": "<<hiddenLayers[j][k].error<<"\n";
		}
	}
	if(debug == 2) std::cout<<"\n=OUTPUT WEIGHTS=\n";
	
	for(int i=0;i<outputLayer.size();++i){
		if(debug == 2) std::cout<<"O"<<i<<":\n";
		for(int j=0;j<outputLayer[i].weights.size();++j){
			if(j!=outputLayer[i].weights.size()-1){
				outputLayer[i].weights[j] -= learnSpeed*outputLayer[i].error*hiddenLayers.back()[j].value;
				if(debug == 2) std::cout<<"  W"<<j<<": "<<outputLayer[i].weights[j]<<"\n";
			}
			else{
				outputLayer[i].weights[j] -= learnSpeed*outputLayer[i].error;
				if(debug == 2) std::cout<<"  B: "<<outputLayer[i].weights[j]<<"\n";
			}
		}
	}
	if(debug == 2) std::cout<<"\n=HIDDEN WEIGHTS=\n";

	for(int i=0;i<hiddenLayers.size();++i){
		if(debug == 2) std::cout<<"L"<<i<<":";
		for(int j=0;j<hiddenLayers[i].size();++j){
			if(debug == 2) std::cout<<"\n  N"<<j<<":\n";
			for(int k=0;k<hiddenLayers[i][j].weights.size();++k){
				if(k!=hiddenLayers[i][j].weights.size()-1){
					float input;
					if(i==0){
						input = tanh(inputs[k]);
					}
					else{
						input = hiddenLayers[i-1][k].value;
					}
					hiddenLayers[i][j].weights[k] -= learnSpeed*hiddenLayers[i][j].error*input;
					if(debug == 2) std::cout<<"    W"<<k<<": "<<hiddenLayers[i][j].weights[k];;
				}
				else{
					hiddenLayers[i][j].weights[k] -= learnSpeed*hiddenLayers[i][j].error;
					if(debug == 2) std::cout<<"    B: "<<hiddenLayers[i][j].weights[k];
				}
			}
		}
	}


	return output;
}

void AINet::trainNet(std::vector<std::vector<float>> inputs,std::vector<std::vector<float>> expected,int dataCount,int iterations,float learnSpeed,int debugLevel){
	std::vector<float> output;

	if(debugLevel==1) std::cout<<"\033[34m\n\n=========================TRAINING=========================\n\n\n\n==========================================================\033[2A";
	

	int totalCorrect = 0;
	int recentCorrect = 0;
	int recentCount = 1000;
	std::vector<bool> lastRecent;

	for(int i=0;i<iterations;++i){
		int index = i%(dataCount);
		output = trainNet(inputs[index],expected[index],learnSpeed,debugLevel);
		float max = -999;
		int outputIndex;
		int expectedIndex;
		for(int j=0;j<10;++j){
			if(output[j]>max){
				max = output[j];
				outputIndex = j;
			}
			if(expected[index][j]==1){
				expectedIndex=j;
			}
		}
		if(expectedIndex==outputIndex){
			totalCorrect++;
			recentCorrect++;
		}

		lastRecent.push_back(expectedIndex==outputIndex);

		if(i>recentCount){
			if(lastRecent[0]){
				recentCorrect--;
			}
			auto it = lastRecent.begin();
			lastRecent.erase(it);
		}

		if(debugLevel==1) std::cout<<"\r\033[93mexpected: \033[35m"<<expectedIndex<<"\033[93m | actual: \033[35m"<<outputIndex<<"\033[93m | correct: \033[32m"<<totalCorrect<<"\033[93m | \033[36m"<<"total: \033[36m"<<i+1<<"\033[93m | \033[36m"<<((float)recentCorrect*(i<recentCount?100:1)/(i<recentCount?i+1:recentCount/100))<<"%    ";
	}
}

void AINet::trainNet(std::string dataFilePath,int dataCount,int iterations,float learnSpeed,int debugLevel){
	std::vector<std::vector<float>> inputs = getInputs(dataFilePath,dataCount);
	std::vector<std::vector<float>> expected = getExpected(dataFilePath,outputLayer.size(),dataCount);
	trainNet(inputs,expected,dataCount,iterations,learnSpeed,debugLevel);
}