#include "NeuralNetwork.h"
#include <ctime>

int main() {
	srand(time(0)*1000); 
	
	NeuralNetwork net;

	net.randomize();

	std::vector<double> input = { 0.9f,0.7f,0.5f };

	net.query(input);

	return true;
}