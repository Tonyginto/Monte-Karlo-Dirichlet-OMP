#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cmath>
#include <vector>
#include <omp.h>

#define THREADS 4

using namespace std;

int allRandom(unsigned int &seed);
float RndKoeff(vector<vector<float>> ar, int a, int b, int an, int bn, unsigned int &seed);
float g(int a, int b);
vector<vector<float>> FillArr(int an, int bn);
float MK(int an, int bn, int N, int i, int j, vector<vector<float>> ar);

float MK_OML(int an, int bn, int N, int i, int j, vector<vector<float>> ar);

int main()
{
	int an = 10, bn = 10, in = 5, jn = 5;
	long int N = 1000000;

	vector<vector<float>> ar = FillArr(an, bn);

	// Последовательное время
	/*double fbg = clock();
	float solMK = MK(an, bn, N, in, jn, ar);

	cout << "Sequential time execution: " << (clock() - fbg) / 1000.0 << endl;
	cout << "Sequential solution at point [" << in << "," << jn << "]: " << solMK << "\n\n";*/

	// Параллельное время

	double sbg = clock();
	float solMK_OMP = MK_OML(an, bn, N, in, jn, ar);

	cout << "Parallel time execution: " << (clock() - sbg) / 1000.0 << endl;
	cout << "Parallel solution at point [" << in << "," << jn << "]: " << solMK_OMP << "\n\n";

	// cout << "Difference of solutions at point [" << in << "," << jn << "]: " << abs(solMK - solMK_OMP) << endl;
}

int allRandom(unsigned int &seed) {
	seed = (8253729 * seed + 2396403);

	// Берем стартовое число и возвращаем значение в диапазоне от 0 до 3
	return (seed * 32768) / 32767 % 4;

}

float RndKoeff(vector<vector<float>> ar, int a, int b, int an, int bn, unsigned int &seed) {
	while (a != 0 && a != an - 1 && b != 0 && b != bn - 1) {
		int rnd = allRandom(seed);
		if (rnd == 0) {
			a--;
		}
		if (rnd == 1) {
			a++;
		}
		if (rnd == 2) {
			b--;
		}
		if (rnd == 3) {
			b++;
		}
	}
	return ar[a][b];
}

float g(int a, int b) {
	return sin(a+ 3.1415 / 2) * sin(3*b + 3.1415 / 2);
}

vector<vector<float>> FillArr(int an, int bn) {
	vector<vector<float>> ar(an, vector <float>(bn));
	for (int i = 0; i < an; i++) {
		for (int j = 0; j < bn; j++) {
			if (i != 0 && i != an - 1 && j != 0 && j != bn - 1) {
				ar[i][j] = 0;
			}
			else {
				ar[i][j] = g(i, j);
			}
		}
	}
	return ar;
}

float MK(int an, int bn, int N, int i, int j, vector<vector<float>> ar) {

	float rnk = 0;
	unsigned int seed = 1563;
	for (long int k = 0; k < N; k++) {
		rnk += RndKoeff(ar, i, j, an, bn, seed);
	}
	rnk = rnk / N;

	return rnk;
}

float MK_OML(int an, int bn, int N, int i, int j, vector<vector<float>> ar) {
	omp_set_num_threads(THREADS);
	float rnk = 0;

#pragma omp parallel
	{
		int thread = omp_get_thread_num();
		int amount = omp_get_num_threads();
		unsigned int seed = thread + 1563;
# pragma omp for reduction (+:rnk)
		for (long int k = 0; k < N; k++) {
			rnk += RndKoeff(ar, i, j, an, bn, seed);
		}
	}
	rnk = rnk / N;

	return rnk;
}