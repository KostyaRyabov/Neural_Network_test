#include "NeuralNetwork.h"
#include <ctime>

int main() {
	srand(time(0)*1000); 
	
	NeuralNetwork net;

	net.randomize();

	std::vector<double> input = { 0.9f,0.7f,0.5f };
	std::vector<double> target = { 0.55f,0.61f,0.15f };

	net.train(input, target);

	//net.query(input);

	return true;
}