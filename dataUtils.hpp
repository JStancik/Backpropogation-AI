#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <C:\Users\Stancik Family\GitHub\Backpropogation-AI\debugOutput.hpp>

inline std::vector<std::vector<float>> getInputs(std::string filename,int lineCount,int skipped = 1){
	std::string line;
	std::ifstream file;
	file.open(filename);

	std::getline(file,line);

	std::vector<std::vector<float>> inputs;
	debug::loadBar loadBar("LOADING INPUTS:",20,lineCount,"\033[34m","\033[32m");
	for(int i=0;i<lineCount;++i){
		loadBar.incrLoad(i);
		std::getline(file,line);
		for(int j=0;j<skipped;++j){
			line.erase(0,line.find(',')+1);
		}
		std::vector<float> input;
		while(line.find(',')!=line.npos){
			std::string sVal = line.substr(0,line.find(','));
			float val = std::stof(sVal);
			input.push_back(val);
			line.erase(0,line.find(',')+1);
		}
		input.push_back(std::stof(line));
		inputs.push_back(input);
	}
	file.close();
	return inputs;
}

inline std::vector<std::vector<float>> getExpected(std::string filename,int count,int lineCount,int skipped = 0){
	std::ifstream file;
	file.open(filename);
	std::string line;
	std::getline(file,line);
	std::cout<<"\n\n";
	debug::loadBar loadBar("LOADING VALUES:",20,lineCount,"\033[34m","\033[32m");

	std::vector<std::vector<float>> inputs;
	for(int i=0;i<lineCount;++i){
		loadBar.incrLoad(i);
		std::getline(file,line);
		std::vector<float> input;
		for(int j=0;j<skipped;++j){
			line.erase(0,line.find(',')+1);
		}
		for(int j=0;j<count;++j){
			input.push_back(0);
		}
		input[std::stoi(line.substr(0,line.find(',')))]=1;
		inputs.push_back(input);
	}
	file.close();
	return inputs;
}