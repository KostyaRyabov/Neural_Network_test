#pragma once

#include <amp.h>
#include <amp_math.h>

#include <fstream>
#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>

/*
N - кол-во слоев

один шаг:
1. запрос gpu обработать N-1 раз X=W*I
2. запрос gpu обработать N-1 раз dW=[E*S(1-S)]*[O]
3. запрос gpu обработать N-1 раз W+=dW
*/

using namespace concurrency;

const unsigned int lc = 3;
const unsigned int amountOfNodesOnLayers[lc] = { 3,5,3 };

#define learning_rate 0.3f



#define sygm(val) (1 / (1 + fast_math::exp(val)))


class NeuralNetwork
{
public:
	//NeuralNetwork(const char* filename);			// берем из файла структуру нейронной сети
	
	NeuralNetwork();
	~NeuralNetwork();								// сохраняем проделанное в файл

	void init();
	void train(std::vector<double> & input, std::vector<double> &target);
	std::vector<double> query(std::vector<double> &input);

	void randomize();
private:
	unsigned int nodes_sum = 0;

	std::vector<std::vector<double>> O_data;
	std::vector<std::vector<double>> E_data;
	std::vector<std::vector<double>> W_data;

	std::vector<concurrency::array_view<double, 2>> W_av;
};