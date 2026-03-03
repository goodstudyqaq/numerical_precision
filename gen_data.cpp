#include <iostream>
#include <iomanip>
#include <random>
using namespace std;

int main() {
    freopen("data.in", "w", stdout);
    int n = 1000000;
    cout << n << endl;

    double X = 1e5;


    mt19937_64 gen(random_device{}());

    uniform_real_distribution<double> small(-X, X);


    for (int i = 0; i < n; i++) {
        double v = small(gen);
        cout << fixed << setprecision(10) << v << endl;
    }
}