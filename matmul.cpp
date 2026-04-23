#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstring>

// Matrix multiply: C = A * B, all NxN
void matmul(const double* A, const double* B, double* C, int N) {
    memset(C, 0, N * N * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i*N+j] += A[i*N+k] * B[k*N+j];
}

int main(int argc, char* argv[]) {
    int N = 512;
    int RUNS = 5;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) RUNS = atoi(argv[2]);

    std::vector<double> A(N*N), B(N*N), C(N*N);
    // Initialize with deterministic values
    for (int i = 0; i < N*N; i++) {
        A[i] = (i % 100) * 0.01 + 1.0;
        B[i] = ((i*7 + 3) % 100) * 0.01 + 1.0;
    }

    // Warmup
    matmul(A.data(), B.data(), C.data(), N);

    // Timed runs
    double total = 0.0;
    for (int r = 0; r < RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        matmul(A.data(), B.data(), C.data(), N);
        auto end = std::chrono::high_resolution_clock::now();
        total += std::chrono::duration<double>(end - start).count();
    }

    double mean = total / RUNS;
    // Print checksum to prevent dead code elimination
    double checksum = 0;
    for (int i = 0; i < N*N; i++) checksum += C[i];
    std::cout << mean << " " << checksum << std::endl;
    return 0;
}
