#include<bits/stdc++.h>
using namespace std;

// float is single precision, double is double precision.
#define real double

void calculate_sum(const vector<real> &data, real &sum) {
    // Use 1000 threads to calculate the sum in parallel.
    int n = data.size();
    int num_threads = min(1000, n);
    sum = 0;
    std::mutex mtx; // mutex for critical section

    int block = n / num_threads;
    vector<thread> threads;
    for (int i = 0; i < num_threads; i++) {
        threads.emplace_back([&, i]() {
            int start = i * block;
            int end = (i == num_threads - 1) ? n : start + block;
            real local_sum = 0;
            for (int j = start; j < end; j++) {
                local_sum += data[j];
            }
            lock_guard<std::mutex> lock(mtx); // lock the mutex before updating the sum
            sum += local_sum;
        });
    }
    for (auto &t : threads) {
        t.join();
    }
}


void calculate_sum_sequential(const vector<real> &data, real &sum) {
    sum = 0;
    for (const auto &x : data) {
        sum += x;
    }
}

void calculate_sum_fixed_point(const vector<real> &data, real &sum) {
    vector<long long> fixed_data(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        fixed_data[i] = static_cast<long long>(data[i] * 0x100000000); // Scale to fixed-point representation
    }

    long long fixed_sum = 0;
    for (const auto &x : fixed_data) {
        fixed_sum += x;
    }
    sum = static_cast<real>(fixed_sum) / 0x100000000; // Scale back to floating-point representation
}


int main() {
    freopen("data.in", "r", stdin);
    int n;
    cin >> n;
    vector<real> data(n);
    for (int i = 0; i < n; i++) {
        real x;
        cin >> x;
        data[i] = x;
    }
    real sum;
    for (int t = 0; t < 10; t++) {
        sum = 0;
        calculate_sum_sequential(data, sum);
        cout << "Test " << t + 1 << ": Sequential sum = " << fixed << setprecision(10) << sum << endl;

    }
    for (int t = 0; t < 10; t++) {
        sum = 0;
        calculate_sum(data, sum);
        cout << "Test " << t + 1 << ": Parallel(threads = 1000) sum = " << fixed << setprecision(10) << sum << endl;
    }

    for (int t = 0; t < 10; t++) {
        sum = 0;
        calculate_sum_fixed_point(data, sum);
        cout << "Test " << t + 1 << ": Fixed-point sum = " << fixed << setprecision(10) << sum << endl;
    }

    return 0;
}