#pragma once
#include <random>
#include "network.hpp"
#include "dataUtils.hpp"

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
	debugTime = clock();
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

void AINet::backPropogate(std::vector<float> &inputs,std::vector<float> &output,std::vector<float> &expected,float &learnSpeed){
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

	for(int i=0;i<outputLayer.size();++i){
		outputLayer[i].error = (outputLayer[i].value-expected[i])*tanh_derivative(outputLayer[i].value);
	}
	clearHiddenLayer();
	for(int j=hiddenLayers.size()-1;j>=0;--j){
		std::vector<Neuron>* forwardLayer = j==hiddenLayers.size()-1?&outputLayer:&hiddenLayers[j+1];
		for(int k=0;k<hiddenLayers[j].size();++k){
			
			for(int i=0;i<(*forwardLayer).size();++i){
				hiddenLayers[j][k].error += (*forwardLayer)[i].error*(*forwardLayer)[i].weights[k];
			}
		}
		for(int k=0;k<hiddenLayers[j].size();++k){
				hiddenLayers[j][k].error *= tanh_derivative(hiddenLayers[j][k].value);
		}
	}
	
	for(int i=0;i<outputLayer.size();++i){
		for(int j=0;j<outputLayer[i].weights.size();++j){
			if(j!=outputLayer[i].weights.size()-1){
				outputLayer[i].weights[j] -= learnSpeed*outputLayer[i].error*hiddenLayers.back()[j].value;
			}
			else{
				outputLayer[i].weights[j] -= learnSpeed*outputLayer[i].error;
			}
		}
	}

	for(int i=0;i<hiddenLayers.size();++i){
		for(int j=0;j<hiddenLayers[i].size();++j){
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
				}
				else{
					hiddenLayers[i][j].weights[k] -= learnSpeed*hiddenLayers[i][j].error;
				}
			}
		}
	}
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

	debugTime = clock();
	forwardPropTime = debugTime - lastTimeStep;

	backPropogate(inputs,output,expected,learnSpeed);
	
	debugTime = clock();
	BackwardPropTime = debugTime-forwardPropTime-lastTimeStep;
	return output;
}

void AINet::trainNet(std::vector<std::vector<float>> inputs,std::vector<std::vector<float>> expected,int dataCount,int epochs,float learnSpeed,int debugLevel){
	std::vector<float> output;

	if(debugLevel>0) std::cout<<"\033[34m\n\n=========================TRAINING=========================\n\n\n\n";
	if(debugLevel>1) std::cout<<            "=========================TIMELINE=========================\n\n\n\n";
	if(debugLevel>0) std::cout<<            "==========================================================";
	if      (debugLevel==2) std::cout<<"\033[6A";
	else if (debugLevel==1) std::cout<<"\033[2A";

	miscTime = 0;
	int totalCorrect = 0;
	int recentCorrect = 0;
	int recentCount = 1000;
	std::vector<bool> lastRecent;

	for(int i=0;i<epochs*dataCount;++i){
		debugTime = clock();
		lastTimeStep = debugTime;
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
		debugTime = clock();

		if(debugLevel>0) std::cout<<"\r\033[93m expected: \033[35m"<<expectedIndex<<"\033[93m | actual: \033[35m"<<outputIndex<<"\033[93m | correct: \033[32m"<<totalCorrect<<"\033[93m | \033[36m"<<"total: \033[36m"<<i+1<<"\033[93m | \033[36m"<<((float)recentCorrect*(i<recentCount?100:1)/(i<recentCount?i+1:recentCount/100))<<"%      ";
		if(debugLevel>1) std::cout<<"\033[4B\r\033[31m forward: "<<forwardPropTime<<"ms | backward: "<<BackwardPropTime<<"ms | misc: "<<miscTime<<"ms\033[4A";
		miscTime = debugTime-BackwardPropTime-forwardPropTime-lastTimeStep;
	}
}

void AINet::trainNet(std::string dataFilePath,int dataCount,int iterations,float learnSpeed,int debugLevel){
	std::vector<std::vector<float>> inputs = getInputs(dataFilePath,dataCount);
	std::vector<std::vector<float>> expected = getExpected(dataFilePath,outputLayer.size(),dataCount);
	trainNet(inputs,expected,dataCount,iterations,learnSpeed,debugLevel);
}