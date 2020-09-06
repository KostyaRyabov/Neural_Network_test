#pragma once

#include <amp.h>
#include <amp_math.h>

#include <stdexcept>
#include <vector>
#include <random>
#include <iostream>
#include <fstream>

/*
N - ���-�� �����

���� ���:
1. ������ gpu ���������� N-1 ��� X=W*I
2. ������ gpu ���������� N-1 ��� dW=[E*S(1-S)]*[O]
3. ������ gpu ���������� N-1 ��� W+=dW
*/

using namespace concurrency;

const unsigned int lc = 3;
const unsigned int amountOfNodesOnLayers[lc] = { 784,100,10 };

#define learning_rate 0.2f



#define sygm(val) (1 / (1 + fast_math::exp(val)))


class NeuralNetwork
{
public:
	//NeuralNetwork(const char* filename);			// ����� �� ����� ��������� ��������� ����
	
	NeuralNetwork();
	~NeuralNetwork();								// ��������� ����������� � ����

	void init();
	void train(std::vector<double> &input, std::vector<double> &target);
	std::vector<double> query(std::vector<double> &input);

	void randomize();

	void saveData();
	void loadData(const char* filename);
private:
	unsigned int nodes_sum = 0;

	std::vector<std::vector<double>> O_data;
	std::vector<std::vector<double>> E_data;
	std::vector<std::vector<double>> W_data;

	std::vector<concurrency::array_view<double, 2>> W_av;
};