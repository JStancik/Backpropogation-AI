#include "network.cpp"

int main(){
	std::cout<<"\033[?25l";
	srand(5);
	std::vector<std::vector<float>> inputs = getInputs("train.csv",42000);
	std::vector<std::vector<float>> expected = getExpected("train.csv",10,42000);
	
	std::cout<<"\n\n\n";
	AINet network(784,4,256,10);

	network.trainNet(inputs,expected,41900,500000,0.002f,1);
	
	std::cout<<"\n\n\n";
	std::vector<float> output;
	int right = 0;
	int wrong = 0;
	int total = 100;
	for(int i=42000-total;i<42000;++i){
		output=network.getOutput(inputs[i]);
		float max = -INFINITY;
		int outputIndex;
		int expectedIndex;
		for(int j=0;j<10;++j){
			if(output[j]>max){
				max = output[j];
				outputIndex = j;
			}
			if(expected[i][j]==1){
				expectedIndex=j;
			}
		}
		if(outputIndex==expectedIndex){
			std::cout<<"\033[32m";
			right++;
		}
		else{
			std::cout<<"\033[31m";
			wrong++;
		}
		std::cout<<"Trial "<<i-41900<<": \033[m"<<"guess: \033[95m"<<outputIndex<<"\033[36m | \033[mactual: \033[95m"<<expectedIndex<<std::endl;
	}

	std::cout<<"\n\033[93mcorrect: "<<right<<" | incorrect: "<<wrong<<" | score: "<<(float)right/total*100<<"%\033[m";
	return 0;
}