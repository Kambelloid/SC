#include <iostream>
#include <chrono>
#include <mpi.h>
#include <cmath>

//#pragma once
//using namespace std::chrono;
//class Time
//{
//public:
//    Time() = default;
//    void start() {
//        start_point = high_resolution_clock::now();
//    }
//
//    void stop() {
//        end_point = high_resolution_clock::now();
//        auto start = time_point_cast<microseconds>(start_point).time_since_epoch().count();
//        auto end = time_point_cast<microseconds>(end_point).time_since_epoch().count();
//        std::cout << "Time taken = " << (end - start) << " microseconds\n";
//    }
//
//private:
//    time_point<high_resolution_clock> start_point;
//    time_point<high_resolution_clock> end_point;
//};

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

double get_disc(int N, double** matrix_A, double* x_k, double const* vector_f)
{
    double disc = 0;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            disc += matrix_A[i][j] * x_k[j];
        disc -= vector_f[i];
    }

    return disc;
}

int Yakobi_Seidel_method(int N, double eps_1, double eps_2, double** matrix_A, double const* vector_f, double* x_0, double* x_k, int rank, int size)
{
    auto parts_size = new int [size + 1];
    parts_size[0] = 0;
    for (int i = 0; i < size; i++) {
        parts_size[i + 1] = parts_size[i] + (N >= size ? N / size + (rank < N % size ? 1 : 0) : 1);
//        std::cout << parts_size[i + 1] << '\t';
    }
//    std::cout << std::endl;

    int cur_start = parts_size[rank];
    int cur_stop = parts_size[rank + 1];
    int cur_size = parts_size[rank + 1] - parts_size[rank];
    for (int i = 0; i < N; i++)
        x_k[i] = vector_f[i] / matrix_A[i][i];

    int iter = 0;
    double discrepancy = eps_1 + 1;
    double disc1 = 0;
    double disc2 = 0;
    double global_disc = 0;
    bool expression = true;

    while (discrepancy > eps_1 || expression) {
        for (int i = cur_start; i < cur_stop; i++) {
            x_0[i] = x_k[i];
            for (int j = 0; j < N; j++)
                disc1 += matrix_A[i][j] * x_k[j];

            disc1 -= vector_f[i];
            disc2 += pow(disc1, 2);

            disc1 = 0;
        }

        MPI_Allreduce(&disc2, &global_disc, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        discrepancy = sqrt(global_disc / N);
        disc2 = 0;
        for (int i = cur_start; i < cur_stop; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j)
                    continue;

                disc2 += matrix_A[i][j] * x_0[j];
            }

            x_k[i] = (vector_f[i] - disc2) / matrix_A[i][i];

            disc2 = 0;
        }

        expression = false;
        for (int i = cur_start; i < cur_stop; i++) {
            if (abs(x_0[i] - x_k[i]) > eps_2) {
                expression = true;
                break;
            }
        }
        MPI_Bcast(&expression, 1, MPI_CXX_BOOL, 0, MPI_COMM_WORLD);
        iter++;
//        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, x_k, cur_size, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DOUBLE, x_0, cur_size, MPI_DOUBLE, MPI_COMM_WORLD);
    }

//    -0.0995126	0.845975	0.906939	0.125234	-0.78029	-0.979691	-0.290892	0.653999

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

//    for (int i = 0; i < N; i++)
//        std::cout << x_k[i] << ' ';
//    std::cout << std::endl;

//    MPI_Barrier(MPI_COMM_WORLD);
//    auto res = new double [N];
//    MPI_Barrier(MPI_COMM_WORLD);

    return iter;
}


//int main()
//{
//    try {
//        omp_set_dynamic(1);
//        int cores[] = {1, 2, 3, 4, 5, 6, 7, 8};
//        int sizes[] = {500, 1000, 5000, 10000, 15000, 20000};
//        double eps_1 = 1e-6, eps_2 = 1e-6;
//        for(auto size : sizes) {
//            std::cout << "Duration for N = " << size << ":" << std::endl;
//            for(auto core : cores) {
//                omp_set_num_threads(core);
//                double **matrix = new double *[size];
//                for (int i = 0; i < size; i++)
//                    matrix[i] = new double[size];
//                double *vector_f = new double[size];
//
//                for (int i = 0; i < size; i++) {
//                    for (int j = 0; j < size; j++)
//                        matrix[i][j] = i == j ? 1 : 0.1 / (i + j);;
//                    vector_f[i] = sin(i);
//                }
//
//                auto start = omp_get_wtime();
//                Yakobi_Seidel_method(size, eps_1, eps_2, matrix, vector_f);
////              std::cout << "Yakobi and Seidel methods completed successfully!";
//                auto end = omp_get_wtime();
//                std::cout << end - start << '\t';
//
//                delete[] matrix;
//                delete[] vector_f;
//            }
//            std::cout << std::endl;
//        }
//
////        for(int i = 0; i < N; i++) {
////            for (int j = 0; j < N; j++)
////                std::cout << matrix[i][j] << '\t';
////            std::cout << '\t' << vector_f[i] << std::endl;
////        }
//    }
//
//    catch (const char* exception) {
//        std::cout << exception;
//        EXIT_FAILURE;
//    }
//}


int main(int argc, char **argv) {

    int N = 10000; double eps_1 = 1e-7, eps_2 = 1e-7;
    auto *x_0 = new double[N];
    auto *x_k = new double[N];

    MPI_Init(&argc, &argv);

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&eps_2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x_0, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x_k, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto** matrix = new double*[N];
    for (int i = 0; i < N; i++)
        matrix[i] = new double[N];
    auto* f_coef = new double[N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            matrix[i][j] = i == j ? 1 : 0.1 / (i + j);
        f_coef[i] = sin(i);
    }

    auto start = MPI_Wtime();
    Yakobi_Seidel_method(N, eps_1, eps_2, matrix, f_coef, x_0, x_k, rank, size);
    auto stop = MPI_Wtime();

    MPI_Finalize();
    if (rank == 0) {
//        for (int i = 0; i < N; i++)
//            std::cout << x_k[i] << '\t';
//        std::cout << std::endl;
        std::cout << N << std::endl;
        std::cout << get_disc(N, matrix, x_k, f_coef) << std::endl;
        std::cout << stop - start << std::endl;
    }

    delete [] matrix;
    delete [] f_coef;
    return 0;
}