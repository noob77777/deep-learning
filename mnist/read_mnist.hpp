#ifndef READ_MNIST
#define READ_MNIST

#include <fstream>
#include <vector>
#include <iostream>

using namespace std;

vector<vector<float>> get_images(string filename, int n) {
	vector<vector<float>> images;
	ifstream ifs(filename);
	ifs.seekg(16, ios::beg);

	char * buffer = new char[28 * 28];
	for(int i = 0; i < n; i++) {
		ifs.read(buffer, 28 * 28);
		assert(ifs);

		vector<float> img;
		for(int j = 0; j < 28 * 28; j++) {
			float tmp = ((static_cast<unsigned int> (buffer[j])) >> 24);
			tmp /= 255.0;
			img.push_back(tmp);
		}
		images.push_back(img);
	}

	delete[] buffer;
	ifs.close();

	return images;
}

vector<vector<float>> get_labels(string filename, int n) {
	vector<vector<float>> labels;
	ifstream ifs(filename);
	ifs.seekg(8, ios::beg);

	char * buffer = new char[n];
	
	ifs.read(buffer, n);
	assert(ifs);

	for(int i = 0; i < n; i++) {
		vector<float> label(10);
		int idx = static_cast<int> (buffer[i]);
		label[idx] = 1.0;
		labels.push_back(label);
	}

	delete[] buffer;
	ifs.close();

	return labels;
}

void show_image(vector<float> img) {
	for(int i = 0; i < 28; i++) {
		for(int j = 0; j < 28; j++) {
			cout << (img[28 * i + j] > 0.5 ? "@ " : ". ");
		}
		cout << '\n';
	}
}

int get_max_idx(vector<float> label) {
	int idx = -1; float curr = -1;
	for(int i = 0; i < 10; i++) {
		if(label[i] > curr) {
			curr = label[i];
			idx = i;
		}
	}
	return idx;
}


#endif
