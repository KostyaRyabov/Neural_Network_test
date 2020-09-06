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
	O_data.push_back(std::vector<double>(amountOfNodesOnLayers[0]));
	E_data.push_back(std::vector<double>(amountOfNodesOnLayers[0]));

	// инициализация их представлений

	for (unsigned int i = 1; i < lc; i++) {
		nodes_sum += amountOfNodesOnLayers[i];

		O_data.push_back(std::vector<double>(amountOfNodesOnLayers[i]));
		E_data.push_back(std::vector<double>(amountOfNodesOnLayers[i]));

		W_data.push_back(std::vector<double>(amountOfNodesOnLayers[i - 1] * amountOfNodesOnLayers[i]));
		W_av.push_back(concurrency::array_view<double, 2>(amountOfNodesOnLayers[i], amountOfNodesOnLayers[i-1], W_data.back().data()));
	};
}

void NeuralNetwork::train(std::vector<double>& input, std::vector<double>& target) {
	E_data[lc-1] = query(input);

	// нахождение выходной ошибки
	{
		concurrency::array_view<double, 2> E(E_data[lc - 2].size(), 1, E_data[lc - 1].data());
		concurrency::array_view<double, 2> T(target.size(), 1, target.data());

		parallel_for_each(E.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				E[idx] -= T[idx];
			});

		E.synchronize();
	}

	// транспонирование матрицы весов и умножение ее на ошибку на каждом слое
	for (unsigned int i = lc - 1; i >= 1; i--) {
		concurrency::array_view<double, 2> E_1(E_data[i-1].size(), 1, E_data[i-1]);
		concurrency::array_view<double, 2> E_2(E_data[i].size(), 1, E_data[i]);

		auto &W = W_av[i-1];
		unsigned int count = W.extent[0], size = W.extent[1];

		double *sum_w = new double[size];
		concurrency::array_view<double, 2> sW(size, 1, sum_w);

		// поиск суммы весов

		parallel_for_each(sW.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int col = idx[0];
				sW(idx) = 0;
				for (int k = 0; k < count; k++)
					sW(idx) += W(k, col);
			});

		sW.synchronize();


		// нахождение значений ошибок предыдущего слоя

		parallel_for_each(E_1.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int col = idx[0];
				double sum = 0.0f;
				for (int k = 0; k < count; k++)
					sum += W(k, col) / sW[idx] * E_2(k,1);
				E_1[idx] = sum;
			});

		E_1.synchronize();

		// изменение весов

		concurrency::array_view<double, 2> O_1(O_data[i - 1].size(), 1, O_data[i - 1]);
		concurrency::array_view<double, 2> O_2(O_data[i].size(), 1, O_data[i]);

		auto www = W_data[i-1];

		parallel_for_each(W.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int row = idx[0], col = idx[1];

				double O_k = O_2(row, 1);

				W(row, col) += learning_rate * E_2(row, 1) * O_k * (1 - O_k) * O_1(col, 1);
			});

		W.synchronize();

		delete[] sum_w;
	}
}

std::vector<double> NeuralNetwork::query(std::vector<double> &input) {
	O_data[0] = input;

	for (unsigned int i = 0; i < lc - 1; i++) {
		concurrency::array_view<double, 2> X(O_data[i].size(), 1, O_data[i]);
		concurrency::array_view<double, 2> Y(O_data[i+1].size(), 1, O_data[i+1]);

		auto W = W_av[i];

		unsigned int size = W.extent[1];

		parallel_for_each(Y.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int row = idx[0], col = idx[1];
				float sum = 0.0f;
				for (int k = 0; k < size; k++)
					sum += W(row, k) * X(k, col);
				Y[idx] = sygm(sum);
			});

		Y.synchronize();
	}

	return O_data.back();
}

void NeuralNetwork::randomize()
{
	for (auto &w : W_data) {
		for (unsigned int i = 0; i < w.size(); i++) {
			w[i] = (((double)rand()) / RAND_MAX) * pow(nodes_sum, -0.5f);
			if (w[i] < 0.01f) w[i] = 0.01f;
			else if (w[i] > 0.99f) w[i] = 0.99f;
		}
	}
}