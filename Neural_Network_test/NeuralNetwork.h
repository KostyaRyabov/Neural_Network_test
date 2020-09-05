#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <amp.h>

#include <fstream>
#include <stdexcept>
#include <array>
#include <random>
#include <iostream>
#include <iomanip>

/*
N - ���-�� �����

���� ���:
1. ������ gpu ���������� N-1 ��� X=W*I
2. ������ gpu ���������� N-1 ��� dW=[E*S(1-S)]*[O]
3. ������ gpu ���������� N-1 ��� W+=dW
*/

using namespace concurrency;

const unsigned int layers_count = 3;
const unsigned int amountOfNodesOnLayers[layers_count] = { 5, 4, 10 };

const double learning_rate = 0.3f;




class NeuralNetwork
{
public:
	//NeuralNetwork(const char* filename);			// ����� �� ����� ��������� ��������� ����
	
	NeuralNetwork();
	~NeuralNetwork();								// ��������� ����������� � ����

	void init();
	void train();
	std::vector<double> query(std::vector<double>& I);

	void randomize();
private:
	unsigned int nodes_sum = 0, W_data_size = 0;
	std::array<std::vector<double>,layers_count-1> W;

	double sygm(double val);
};

std::vector<double> operator* (std::vector<double>& Ww, std::vector<double>& input);
