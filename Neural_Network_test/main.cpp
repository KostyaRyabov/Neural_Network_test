#include "NeuralNetwork.h"
#include <ctime>

#define AmountOfImages 66666
#define epoch 10
#define ImageSize 784

using namespace std;

const int MAXN = 6e4 + 7;
std::vector<std::vector<double>> image(MAXN);
unsigned int num, magic, rows, cols;
byte label[MAXN];
unsigned int in(ifstream& icin, unsigned int size) {
    unsigned int ans = 0;
    for (int i = 0; i < size; i++) {
        unsigned char x;
        icin.read((char*)&x, 1);
        unsigned int temp = x;
        ans <<= 8;
        ans += temp;
    }
    return ans;
}
void input() {
    ifstream icin;
    icin.open("t10k-images.idx3-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4), rows = in(icin, 4), cols = in(icin, 4);
    for (unsigned int i = 0; i < min(num, AmountOfImages) ; i++) {
        for (unsigned int x = 0; x < 784; x++) {
            image[i].resize(ImageSize);
            image[i][x] = ((double)in(icin, 1) / 255 * 0.99f) + 0.01f;
        }
        
        if (!(i%(min(num, AmountOfImages)/10))) std::cout << "|";
    }

    std::cout << "\n";

    icin.close();
    icin.open("t10k-labels.idx1-ubyte", ios::binary);
    magic = in(icin, 4), num = in(icin, 4);
    for (unsigned int i = 0; i < min(num, AmountOfImages); i++) {
        label[i] = in(icin, 1);

        if (!(i % (min(num, AmountOfImages) / 10))) std::cout << "|";
    }

    std::cout << "\n";
}


int main() {
	srand(time(0)*1000); 
	
    input();

	NeuralNetwork net;

    
    /*
    net.loadData("07-09-2021.ds");
    */
	net.randomize();

    std::vector<double> target = { 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f };
    
    for (unsigned int e = 0; e < epoch; e++) {
        std::cout << "\n ======[" << e << "]====== \n";

        for (unsigned int i = 0; i < min(num, AmountOfImages); i++) {
            if (i > 0) target[label[i - 1]] = 0.01f;
            target[label[i]] = 0.99f;

            net.train(image[i], target);

            
            if (!(i % (min(num, AmountOfImages) / 10))) std::cout << "|";
        }
    }

    net.saveData();

    for (int i = 0; i < 10; i++) {
        auto v = net.query(image[i]);

        std::cout << "\n";
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                std::cout << ((image[i][r * 28 + c] > 0.5f) ? "#" : " ");
            }
            std::cout << "\n";
        }

        std::cout << "\n\n\t" << (int)label[i] << ":\n";
        for (auto& c : v) {
            std::cout << std::fixed << std::setprecision(3) << c << "  ";
        }
    }
    

	return true;
}