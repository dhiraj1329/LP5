#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
#include <limits>

using namespace std;

int main() {
    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    vector<int> arr(n);

    // Generate random integers
    for (int i = 0; i < n; ++i)
        arr[i] = rand() % 10000;

    int minVal = INT_MAX;
    int maxVal = INT_MIN;
    long long sum = 0;

#pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) reduction(+:sum)
    for (int i = 0; i < n; ++i) {
        if (arr[i] < minVal) minVal = arr[i];
        if (arr[i] > maxVal) maxVal = arr[i];
        sum += arr[i];
    }

    double avg = static_cast<double>(sum) / n;

    cout << "\nResults using Parallel Reduction:\n";
    cout << "Minimum: " << minVal << endl;
    cout << "Maximum: " << maxVal << endl;
    cout << "Sum:     " << sum << endl;
    cout << "Average: " << avg << endl;

    return 0;
}
