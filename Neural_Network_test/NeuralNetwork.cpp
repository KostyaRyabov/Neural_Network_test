#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
	init();
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::init() {
	// инициализация массива весов
	
	nodes_sum = amountOfNodesOnLayers[0];

	// инициализация их представлений

	for (unsigned int i = 1; i < layers_count; i++) {
		nodes_sum += amountOfNodesOnLayers[i];
		W[i-1].resize(amountOfNodesOnLayers[i - 1] * amountOfNodesOnLayers[i]);
		W[i-1].push_back(i);
	};
}

void NeuralNetwork::train() {

}

std::vector<double> NeuralNetwork::query(std::vector<double>& I) {
	std::vector<double> X = W[0] * I;

	for (unsigned int i = 1; i < layers_count-1; i++) {
		X = W[i] * X;
	}

	return X;
}

void NeuralNetwork::randomize()
{
	for (auto &w : W) {
		for (unsigned int i = 0; i < w.size() - 1; i++) {
			w[i] = (((double)rand()) / RAND_MAX) * pow(nodes_sum, -0.5f);
			if (w[i] < 0.01f) w[i] = 0.01f;
		}
	}
}

double NeuralNetwork::sygm(double val) {
	return (1 / (1 + M_E * pow(val, -0.5f)));
}

std::vector<double> operator* (std::vector<double>& Ww, std::vector<double>& input)
{
	int size = input.size(), w_id = (int)Ww.back();

	std::vector<double> output = input;
	array_view<double, 2> I(1, size, input.data());
	array_view<double, 2> W(amountOfNodesOnLayers[w_id-1], amountOfNodesOnLayers[w_id], Ww.data());
	array_view<double, 2> O(1, size, output.data());
	
	parallel_for_each(O.extent,
		[=](concurrency::index<2> idx) restrict(amp) {
			int row = idx[0]; int col = idx[1];
			float sum = 0.0f;
			for (int i = 0; i < size; i++)
				sum += W(row, i) * I(i, col);
			O[idx] = sum;
		});

	return output;
}
