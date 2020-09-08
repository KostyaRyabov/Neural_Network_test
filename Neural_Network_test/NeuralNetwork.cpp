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
	E_data[lc-2] = query(input);

	double sE = 0;
	int ti = -1;
	for (int i = 0; i < target.size(); i++) {
		sE += E_data[lc - 2][i];

		if (target[i] > 0.2) ti = i;
	}

	//std::cout << std::fixed << std::setprecision(1) << "	( " << E_data[lc - 2][ti] / sE*100 << "% )\n";
	
	// нахождение выходной ошибки
	{
		concurrency::array_view<double, 2> E(E_data[lc - 2].size(), 1, E_data[lc - 2].data());
		concurrency::array_view<double, 2> T(target.size(), 1, target.data());

		parallel_for_each(E.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				E[idx] = T[idx] - E[idx];
			});

		E.synchronize();
	}

	for (int i = lc - 2; i > 0; i--) {
		concurrency::array_view<double, 2> E_1(E_data[i - 1].size(), 1, E_data[i - 1]);
		concurrency::array_view<double, 2> E_2(E_data[i].size(), 1, E_data[i]);

		auto& W = W_av[i];
		unsigned int count = W.extent[1], size = W.extent[0];

		// нахождение значений ошибок предыдущего слоя

		parallel_for_each(E_1.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int row = idx[0];
				double sum = 0.0f;
				for (int k = 0; k < size; k++)
					sum += W(k, row)* E_2(k, 0);
				E_1[idx] = sum;
			});

		E_1.synchronize();
	}

	// транспонирование матрицы весов и умножение ее на ошибку на каждом слое
	for (int i = lc - 2; i >= 0; i--) {
		auto& W = W_av[i];
		
		concurrency::array_view<double, 2> E(E_data[i].size(), 1, E_data[i]);
		concurrency::array_view<double, 2> O_1(O_data[i].size(), 1, O_data[i]);
		concurrency::array_view<double, 2> O_2(O_data[i+1].size(), 1, O_data[i+1]);

		parallel_for_each(W.extent,
			[=](concurrency::index<2> idx) restrict(amp) {
				unsigned int row = idx[0], col = idx[1];

				double O_k = O_2(row, 0);

				W(idx) += learning_rate * E(row, 0) * O_k * (1.0f - O_k) * O_1(col,0);
			});

		W.synchronize();
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
				Y[idx] = 1.0f / (1.0f + fast_math::exp(-sum));
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

void NeuralNetwork::saveData()
{
	std::ofstream file("07-09-2021.ds", std::ios::trunc | std::ios::binary);
	if (file) {
		for (unsigned int i = 0; i < lc-1; i++) {
			for (auto& c : W_data[i]) {
				file << c;
			}
		}

		file.close();
	}
}

void NeuralNetwork::loadData(const char* filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (file) {
		std::cout << "\n";
		for (unsigned int i = 0; i < lc - 1; i++) {
			for (auto& c : W_data[i]) {
				file >> c;
			};
			if (!(i % (lc - 1))) std::cout << "*";
		};

		std::cout << "\n";
		file.close();
	}
}
