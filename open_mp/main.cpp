#include <iostream>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <cmath>
#include <iomanip>

#pragma once
using namespace std::chrono;
class Time
{
public:
    Time() = default;
    void start() {
        start_point = high_resolution_clock::now();
    }

    void stop() {
        end_point = high_resolution_clock::now();
        auto start = time_point_cast<microseconds>(start_point).time_since_epoch().count();
        auto end = time_point_cast<microseconds>(end_point).time_since_epoch().count();
        std::cout << "Time taken = " << (end - start) << " microseconds\n";
    }

private:
    time_point<high_resolution_clock> start_point;
    time_point<high_resolution_clock> end_point;
};

//double** fill_matrix(int N, std::ifstream& input)
//{
//    double** matrix = new double* [N];
//    for (int i = 0; i < N; i++)
//        matrix[i] = new double[N];
//
//    for (int i = 0; i < N; i++)
//        for (int j = 0; j < N; j++)
//            input >> matrix[i][j];
//
//    return matrix;
//}
//
//double* fill_f(int N, std::ifstream& input)
//{
//    double* f_coef = new double[N];
//
//    for (int i = 0; i < N; i++)
//        input >> f_coef[i];
//
//    return f_coef;
//}

int Yakobi_Seidel_method(int N, double eps_1, double eps_2, double** matrix_A, double* vector_f)
{
    double* x_0 = new double[N];
    double* x_k = new double[N];
#pragma omp parallel for
    for (int i = 0; i < N; i++)
        x_k[i] = vector_f[i] / matrix_A[i][i];

    int iter = 0;
    double discrepancy = eps_1 + 1;
    double disc1 = 0;
    double disc2 = 0;
    bool expression = true;

    while (discrepancy > eps_1 || expression) {
#pragma omp parallel for reduction(+:disc2) reduction(-:disc1)
        for (int i = 0; i < N; i++) {
            x_0[i] = x_k[i];
#pragma omp parallel for reduction(+ : disc1)
            for (int j = 0; j < N; j++)
                disc1 += matrix_A[i][j] * x_k[j];

            disc1 -= vector_f[i];
            disc2 += pow(disc1, 2);

            disc1 = 0;
        }

        discrepancy = sqrt(disc2 / N);
        disc2 = 0;
#pragma omp parallel for
        for (int i = 0; i < N; i++) {
#pragma omp parallel for reduction(+ : disc2)
            for (int j = 0; j < N; j++) {
                if (i == j)
                    continue;

                disc2 += matrix_A[i][j] * x_0[j];
            }

            x_k[i] = (vector_f[i] - disc2) / matrix_A[i][i];

            disc2 = 0;
        }

        expression = false;
//#pragma omp parallel for
//        for (int i = 0; i < N; i++)
//            if (abs(x_0[i] - x_k[i]) > eps_2)
//                expression = true;
        for (int i = 0; i < N; i++) {
            if (abs(x_0[i] - x_k[i]) > eps_2) {
                expression = true;
                break;
            }
        }

        iter++;
    }

//    std::ofstream output("output_data.txt");
//
//    output << "SLAE solution for matrix" << std::endl << std::endl;
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++)
//            output << matrix_A[i][j] << '\t';
//
//        output << std::endl;
//    }
//
//    output << std::endl << "and vector" << std::endl << std::endl;
//    for (int i = 0; i < N; i++)
//        output << vector_f[i] << '\t';
//    output << std::endl << std::endl;
//
//    output << "is: " << std::endl << std::endl;
//    for (int i = 0; i < N; i++) {
//        if (i == N - 1) {
//            output << "x" << i + 1 << " = " << std::setprecision(16) << x_k[i] << "." << std::endl;
//            continue;
//        }
//
//        output << "x" << i + 1 << " = " << std::setprecision(16) << x_k[i] << ";" << std::endl;
//    }
//
//    output << std::endl << "Number of iterations: " << iter << ".";
//
//    output.close();

    return iter;
}


int main()
{
    try {
        omp_set_dynamic(0);
        int cores[] = {1, 2, 3, 4, 5, 6, 7, 8};
        int sizes[] = {500, 1000, 5000, 10000, 15000, 20000};
        double eps_1 = 1e-6, eps_2 = 1e-6;
        for(auto size : sizes) {
            std::cout << "Duration for N = " << size << ":" << std::endl;
            for(auto core : cores) {
                omp_set_num_threads(core);
                double **matrix = new double *[size];
                for (int i = 0; i < size; i++)
                    matrix[i] = new double[size];
                double *vector_f = new double[size];

                for (int i = 0; i < size; i++) {
                    for (int j = 0; j < size; j++)
                        matrix[i][j] = i == j ? 1 : 0.1 / (i + j);;
                    vector_f[i] = sin(i);
                }

                auto start = omp_get_wtime();
                Yakobi_Seidel_method(size, eps_1, eps_2, matrix, vector_f);
//              std::cout << "Yakobi and Seidel methods completed successfully!";
                auto end = omp_get_wtime();
                std::cout << end - start << '\t';

                delete[] matrix;
                delete[] vector_f;
            }
            std::cout << std::endl;
        }

//        for(int i = 0; i < N; i++) {
//            for (int j = 0; j < N; j++)
//                std::cout << matrix[i][j] << '\t';
//            std::cout << '\t' << vector_f[i] << std::endl;
//        }
    }

    catch (const char* exception) {
        std::cout << exception;
        EXIT_FAILURE;
    }
}

//int main() {
//    /*#pragma omp parallel
//    {
//        #pragma omp for
//        for (int i=0;i<10;i++) printf("%d ", i);
//        #pragma omp single
//        printf("I’m thread %d!\n", omp_get_thread_num());
//        #pragma omp for
//        for (int i=0;i<10;i++) printf("%d ", i);
//    }*/
//    int j;
//    omp_set_num_threads(8);
//#pragma opm parallel shared(j)
//    {
//        std::cout << omp_get_thread_num() << std::endl;
//        for(int i = 0; i < 100; i++)
//            j = i;
//        std::cout << "j = " << j << std::endl;
//
//    }
//    return 0;
//}
