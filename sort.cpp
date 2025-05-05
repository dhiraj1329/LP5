#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace chrono;

// ---------- Sequential Bubble Sort ----------
void bubbleSortSequential(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n-1; ++i)
        for (int j = 0; j < n-i-1; ++j)
            if (arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
}

// ---------- Parallel Bubble Sort using Odd-Even Transposition ----------
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; ++i) {
#pragma omp parallel for
        for (int j = (i % 2); j < n - 1; j += 2) {
            if (arr[j] > arr[j+1]) {
                swap(arr[j], arr[j+1]);
            }
        }
    }
}

// ---------- Sequential Merge Sort ----------
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;

    while (i < left.size() && j < right.size())
        arr[k++] = (left[i] <= right[j]) ? left[i++] : right[j++];

    while (i < left.size()) arr[k++] = left[i++];
    while (j < right.size()) arr[k++] = right[j++];
}

void mergeSortSequential(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSortSequential(arr, l, m);
        mergeSortSequential(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ---------- Parallel Merge Sort ----------
void mergeSortParallel(vector<int>& arr, int l, int r, int depth = 0) {
    if (l < r) {
        int m = (l + r) / 2;

        if (depth <= 4) {
#pragma omp parallel sections
            {
#pragma omp section
                mergeSortParallel(arr, l, m, depth + 1);
#pragma omp section
                mergeSortParallel(arr, m + 1, r, depth + 1);
            }
        } else {
            mergeSortSequential(arr, l, m);
            mergeSortSequential(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

// ---------- Utility ----------
vector<int> generateRandomArray(int size, int maxVal = 10000) {
    vector<int> arr(size);
    for (int& x : arr) x = rand() % maxVal;
    return arr;
}

void printArray(const vector<int>& arr) {
    for (int x : arr) cout << x << " ";
    cout << endl;
}

// ---------- Main ----------
int main() {
    int size;
    cout << "Enter array size: ";
    cin >> size;

    vector<int> arr1 = generateRandomArray(size);
    vector<int> arr2 = arr1;
    vector<int> arr3 = arr1;
    vector<int> arr4 = arr1;

    // Sequential Bubble Sort
    auto start = high_resolution_clock::now();
    bubbleSortSequential(arr1);
    auto end = high_resolution_clock::now();
    cout << "Sequential Bubble Sort Time: " << duration<double, milli>(end - start).count() << " ms\n";

    // Parallel Bubble Sort
    start = high_resolution_clock::now();
    bubbleSortParallel(arr2);
    end = high_resolution_clock::now();
    cout << "Parallel Bubble Sort Time:   " << duration<double, milli>(end - start).count() << " ms\n";

    // Sequential Merge Sort
    start = high_resolution_clock::now();
    mergeSortSequential(arr3, 0, arr3.size() - 1);
    end = high_resolution_clock::now();
    cout << "Sequential Merge Sort Time:  " << duration<double, milli>(end - start).count() << " ms\n";

    // Parallel Merge Sort
    start = high_resolution_clock::now();
#pragma omp parallel
    {
#pragma omp single
        mergeSortParallel(arr4, 0, arr4.size() - 1);
    }
    end = high_resolution_clock::now();
    cout << "Parallel Merge Sort Time:    " << duration<double, milli>(end - start).count() << " ms\n";

    return 0;
}
