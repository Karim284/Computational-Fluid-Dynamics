//
//  main.cpp
//  Project_MCG5157
//
//  Created by Karim Saadeh on 2022-11-01.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SparseCore>
#include <eigen3/Eigen/SparseLU>
#include <eigen3/Eigen/IterativeLinearSolvers>

using namespace std;

template<typename Vector>
void outputResults(int M, int N, string title, vector<double>& x, vector<double>& y, Vector& F)
{
    // writing information to vtk output
    auto filename = title + ".vtk";
    std::ofstream fout(filename);
    fout.precision(16);

    auto num_cells = M * N;

    fout << "# vtk DataFile Version 2.0\n"
        << "Iterative solver\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_GRID\n"
        << "DIMENSIONS " << M << " " << N << " 1\n"
        << "POINTS " << num_cells << " double\n";

    // vector data
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fout << x[i] << " " << y[j] << " 0.0\n";
        }
    }

    // array data
    fout << "\nPOINT_DATA " << num_cells
        << "\nSCALARS " << filename << " double 1"
        << "\nLOOKUP_TABLE default\n";

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fout << F[i][j] << "\n";
        }
    }
}

template<typename Vector>
void updateStarVelocities(Vector& u_Old, Vector& u_Star, Vector& v_Old, Vector& v_Star, double dx, double dy, double dt, double mu, double rho, int M, int N, std::vector<double> x, std::vector<double> y)
{
    double dxi = 1 / dx;
    double dyi = 1 / dy;
    double d2udx2 = 0.0;
    double d2udy2 = 0.0;
    double d2vdx2 = 0.0;
    double d2vdy2 = 0.0;
    double dudx = 0.0;
    double dvdx = 0.0;
    double dudy = 0.0;
    double dvdy = 0.0;
    double rhoi = 1 / rho;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (i == 0)
            {
                u_Old[i][j] = 0.001;
                u_Star[i][j] = 0.001;
                v_Star[i][j] = 0.0;
                v_Old[i][j] = 0.0;
            }
            else if ((y[j] <= (-(x[i] - 1) * (x[i] - 1) + 0.3)) || (y[j] >= ((x[i] - 1) * (x[i] - 1) + 0.7)))
            {
                u_Old[i][j] = 0.0;
                u_Star[i][j] = 0.0;
                v_Star[i][j] = 0.0;
                v_Old[i][j] = 0.0;
            }

            else if (j == 0 || j == N - 1)
            {
                u_Old[i][j] = 0.0;
                u_Star[i][j] = 0.0;
                v_Star[i][j] = 0.0;
                v_Old[i][j] = 0.0;
            }

            else if (i == M - 1)
            {
                d2udx2 = 0.0;
                d2udy2 = (u_Old[i][j + 1] - 2 * u_Old[i][j] + u_Old[i][j - 1]) * dyi * dyi;
                d2vdx2 = 0.0;
                d2vdy2 = (v_Old[i][j + 1] - 2 * v_Old[i][j] + v_Old[i][j - 1]) * dyi * dyi;

                dudx = 0.0;
                dvdx = 0.0;
                dudy = (u_Old[i][j + 1] - u_Old[i][j - 1]) * 0.5 * dyi;
                dvdy = (v_Old[i][j + 1] - v_Old[i][j - 1]) * 0.5 * dyi;

                u_Star[i][j] = u_Old[i][j] + mu * dt * rhoi * (d2udx2 + d2udy2) - dt * (u_Old[i][j] * dudx + v_Old[i][j] * dudy);
                v_Star[i][j] = v_Old[i][j] + mu * dt * rhoi * (d2vdx2 + d2vdy2) - dt * (u_Old[i][j] * dvdx + v_Old[i][j] * dvdy);
            }
            else 
            {
                d2udx2 = (u_Old[i + 1][j] - 2 * u_Old[i][j] + u_Old[i - 1][j]) * dxi * dxi;
                d2udy2 = (u_Old[i][j + 1] - 2 * u_Old[i][j] + u_Old[i][j - 1]) * dyi * dyi;
                d2vdx2 = (v_Old[i + 1][j] - 2 * v_Old[i][j] + v_Old[i - 1][j]) * dxi * dxi;
                d2vdy2 = (v_Old[i][j + 1] - 2 * v_Old[i][j] + v_Old[i][j - 1]) * dyi * dyi;

                dudx = (u_Old[i + 1][j] - u_Old[i - 1][j]) * 0.5 * dxi;
                dvdx = (v_Old[i + 1][j] - v_Old[i - 1][j]) * 0.5 * dxi;
                dudy = (u_Old[i][j + 1] - u_Old[i][j - 1]) * 0.5 * dyi;
                dvdy = (v_Old[i][j + 1] - v_Old[i][j - 1]) * 0.5 * dyi;

                u_Star[i][j] = u_Old[i][j] + mu * dt * rhoi * (d2udx2 + d2udy2) - dt * (u_Old[i][j] * dudx + v_Old[i][j] * dudy);
                v_Star[i][j] = v_Old[i][j] + mu * dt * rhoi * (d2vdx2 + d2vdy2) - dt * (u_Old[i][j] * dvdx + v_Old[i][j] * dvdy);
            }
        }
    }
}

template<typename Vector>
void updateNewVelocities(Vector& u_Star, Vector& u_New, Vector& v_Star, Vector& v_New, Vector& p_New, double dx, double dy, double dt, double rho, int M, int N, std::vector<double> x, std::vector<double> y)
{
    double dxi = 1 / dx;
    double dyi = 1 / dy;
    double rhoi = 1 / rho;

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {

            if (i == 0)
            {
                u_New[i][j] = 0.001;
                v_New[i][j] = 0.0;
            }

            else if ((y[j] <= (-(x[i] - 1) * (x[i] - 1) + 0.3)) || (y[j] >= ((x[i] - 1) * (x[i] - 1) + 0.7)))
            {
                u_New[i][j] = 0.0;
                v_New[i][j] = 0.0;
            }

            else if (j == 0 || j == N - 1)
            {
                u_New[i][j] = 0.0;
                v_New[i][j] = 0.0;
            }

            else 
            {
                u_New[i][j] = u_Star[i][j] - dt * rhoi * (p_New[i][j] - p_New[i - 1][j]) * 0.5 * dxi;
                v_New[i][j] = v_Star[i][j] - dt * rhoi * (p_New[i][j] - p_New[i][j - 1]) * 0.5 * dyi;
            }
        }
    }
}


void buildAMatrix(Eigen::SparseMatrix<double>& A, double dx, double dy, int M, int N)
{
    double dxi = 1 / dx;
    double dyi = 1 / dy;

    double a = 2 / (dx * dx) + 2 / (dy * dy);

    for (int i = 0; i < M * N; i++)
    {
        for (int j = 0; j < N * M; j++)
        {
            if (i == j)
            {
                A.insert(i,j) = - a;
            }

            if (i == (j + 1))
            {
                if (((j + 1) % N) == 0.0)
                {
                    A.insert(i, j) = 0.0;
                }
                else
                {
                    A.insert(i, j) = 1.0 * dxi * dxi;
                }
            }

            if (j == (i + 1))
            {
                if ((j % N) == 0.0)
                {
                    A.insert(i, j) = 0.0;
                }
                else
                {
                    A.insert(i, j) = 1.0 * dxi * dxi;
                }
            }

            if (j == (i + N))
            {
                A.insert(i, j) = 1.0 * dyi * dyi;
            }

            if (i == (j + N))
            {
                A.insert(i, j) = 1.0 * dyi * dyi;
            }
        }
    }
}



void buildbMatrix(Eigen::VectorXd& b, std::vector<std::vector<double>>& u_Star, std::vector<std::vector<double>>& v_Star, double rho, double dt, double dx, double dy, double M, double N)
{
    double dti = 1 / dt;
    double dxi = 1 / dx;
    double dyi = 1 / dy;

    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            auto index = i + M * j;

            if (i == M - 1)
            {
                b(index) = 0.0;

            }

            else
            {
                b(index) = rho * dti * ((u_Star[i + 1][j] - u_Star[i][j]) * 0.5 * dxi + (v_Star[i][j + 1] - v_Star[i][j]) * 0.5 * dyi);
            }  
        }
    }
}



void solvePoissonEquation(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b, Eigen::VectorXd& p)
{
    Eigen::SparseLU<Eigen::SparseMatrix<double>> sparseSolver(A);
    p = sparseSolver.solve(b);
    if (sparseSolver.info() != Eigen::Success) {
        throw std::runtime_error("Linear-system solution failed.");
    }
}



void updatePressure(Eigen::VectorXd& p, std::vector<std::vector<double>>& p_New, int M, int N)
{
    for (int j = 0; j < N; j++)
    {
        for (int i = 0; i < M; i++)
        {
            auto index = i + M * j;
            p_New[i][j] = p(index);
        }
    }
}


template<typename Vector>
double getdT(Vector& p_New, Vector& u_New, Vector& v_New, int M, int N, double dx, double dy)
{
    double dmin = 1.0E6;
    if (dx < dy)
    {
        dmin = dx;
    }
    else 
    {
        dmin = dy;
    }

    double dt = 0.0;
    double CFL = 0.3;
    double max = 0.0;
    double result = 0.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result = abs(u_New[i][j]) + abs(p_New[i][j]) + abs(v_New[i][j]);

            if (max < result)
            {
                max = result;
            }
        }
    }
    dt = CFL * dmin / max;
    return dt;
}

int main(int argc, const char* argv[]) {
    int M = 75, N = 75;
    double L = 5.0, H = 1.0;
    double dx = L / (M - 1);
    double dy = H / (N - 1);
    double t = 75.0;
    double mu = 6; //Blood viscosity [g/(mm.s)]
    double rho = 1060; //Density of blood [g/mm3]

    std::vector<double> x(M);
    std::vector<double> y(N);

    // filling vectors
    for (int i = 0; i < M; ++i) {
        x[i] = i * dx;
    }

    for (int j = 0; j < N; ++j) {
        y[j] = j * dy;
    }

    std::vector<std::vector<double>> u_Old(M, std::vector<double>(N));
    std::vector<std::vector<double>> u_Star(M, std::vector<double>(N));
    std::vector<std::vector<double>> u_New(M, std::vector<double>(N));

    std::vector<std::vector<double>> v_Old(M, std::vector<double>(N));
    std::vector<std::vector<double>> v_Star(M, std::vector<double>(N));
    std::vector<std::vector<double>> v_New(M, std::vector<double>(N));

    std::vector<std::vector<double>> p_New(M, std::vector<double>(N));

    Eigen::SparseMatrix<double> A(M * N, M * N);
    Eigen::VectorXd p(M * N), b(M * N);
    

    double t_end = 0.0;
    int it = 0;
    auto title = "output_" + std::to_string(it);
    double dt = 0.0;

    updateStarVelocities(u_Old, u_Star, v_Old, v_Star, dx, dy, dt, mu, rho, M, N, x, y);
    buildAMatrix(A, dx, dy, M, N);
    dt = getdT(p_New, u_Star, v_Star, M, N, dx, dy);
    buildbMatrix(b, u_Star, v_Star, rho, dt, dx, dy, M, N);
    solvePoissonEquation(A, b, p);
    updatePressure(p, p_New, M, N);
    updateNewVelocities(u_Star, u_New, v_Star, v_New, p_New, dx, dy, dt, rho, M, N, x, y);
    
    outputResults(M, N, title, x, y, u_New);
    
    while (t_end < t)
    {
        if ((1.0 < t_end && t_end < 1.1) || (2.0 < t_end && t_end < 2.1) || (3.0 < t_end && t_end < 3.1) || (5.0 < t_end && t_end < 5.1) || (10.0 < t_end && t_end < 10.1) || (20.0 < t_end && t_end < 20.1) || (50.0 < t_end && t_end < 50.1))
        {
            outputResults(M, N, title, x, y, u_New);
        }
        dt = getdT(p_New, u_New, v_New, M, N, dx, dy);
        it += 1;
        title = "output_" + std::to_string(it);
        t_end = t_end + dt;
        std::cout << t_end << std::endl;
        updateStarVelocities(u_Old, u_Star, v_Old, v_Star, dx, dy, dt, mu, rho, M, N, x, y);
        buildbMatrix(b, u_Star, v_Star, rho, dt, dx, dy, M, N);
        solvePoissonEquation(A, b, p);
        updatePressure(p, p_New, M, N);
        updateNewVelocities(u_Star, u_New, v_Star, v_New, p_New, dx, dy, dt, rho, M, N, x, y);

        u_Old = u_New;
        v_Old = v_New;
    }
    outputResults(M, N, title, x, y, u_New);
}