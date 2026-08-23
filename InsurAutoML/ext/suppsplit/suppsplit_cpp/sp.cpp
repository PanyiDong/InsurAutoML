/*
 * File Name: sp.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)
 *
 * Project: src
 * Latest Version: <<projectversion>>
 * Relative Path: /sp.cpp
 * File Created: Thursday, 11th September 2025 2:44:42 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * Last Modified: Monday, 15th September 2025 2:19:05 pm
 * Modified By: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * MIT License
 *
 * Copyright (c) 2025 - 2025, Panyi Dong
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// sp.cpp
// Plain C++ / Armadillo version of original sp.cpp (Rcpp removed)
//
// Requires: Armadillo, optionally OpenMP
//
#include <vector>
#include <iostream>
#include <cmath>
#include <vector>
#include <float.h>
#include <random>
#include <chrono>

int parallel_threads = 1;
#ifdef _OPENMP
#include <omp.h>
#endif

void printProgress(int percent)
{
    if (parallel_threads == 1)
        std::cout << "\rOptimizing <1 thread> [" << std::string(percent / 5, '+') << std::string(100 / 5 - percent / 5, ' ') << "] " << percent << "%";
    else
        std::cout << "\rOptimizing <" << parallel_threads << " threads> [" << std::string(percent / 5, '+') << std::string(100 / 5 - percent / 5, ' ') << "] " << percent << "%";
    std::cout.flush();
}

std::vector<std::vector<double>> sp_cpp(std::size_t des_num, int dim_num,
                                        const std::vector<std::vector<double>> &ini,
                                        const std::vector<std::vector<double>> &distsamp,
                                        bool thin,
                                        const std::vector<std::vector<double>> &bd,
                                        std::size_t point_num,
                                        int it_max,
                                        double tol,
                                        int num_proc,
                                        double n0,
                                        const std::vector<double> &wts,
                                        bool rnd_flg)
{
#ifdef _OPENMP
    omp_set_num_threads(num_proc);
    parallel_threads = num_proc;
#endif

    int it_num = 0;
    bool cont = true;

    std::vector<double> curconst(des_num, 0.0);
    std::vector<double> runconst(des_num, 0.0);
    std::vector<double> runconst_up(des_num, 0.0);

    // store designs as matrix des (des_num x dim_num)
    // define prevdes, des, des_up as (des_num x dim_num) matrices
    std::vector<double> prevdes(des_num * dim_num);
    std::vector<double> des(des_num * dim_num);
    std::vector<double> des_up(des_num * dim_num);
    for (std::size_t i = 0; i < des_num; i++)
    {
        for (int j = 0; j < dim_num; j++)
        {
            des[i * dim_num + j] = ini[i][j];
        }
    }

    // RNG
    std::default_random_engine generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    double nug = 0.0;
    int percent_complete = 0;
    while (cont)
    {
        int percent = (100 * (it_num + 1)) / it_max;
        if (percent > percent_complete)
        {
            printProgress(percent);
            percent_complete = percent;
        }

        std::fill(curconst.begin(), curconst.end(), 0.0);
        bool nanflg = false;
        prevdes = des;

        // prepare rnd sample and weights
        std::vector<std::vector<double>> rnd(point_num, std::vector<double>(dim_num, 0.0));
        std::vector<double> rnd_wts(point_num, 0.0);
        std::uniform_int_distribution<int> uddist(0, (int)distsamp.size() - 1);
        for (std::size_t i = 0; i < point_num; i++)
        {
            std::size_t ss;
            if (rnd_flg)
                ss = uddist(generator);
            else
                ss = i;

            for (int j = 0; j < dim_num; j++)
                rnd[i][j] = distsamp[ss][j];

            rnd_wts[i] = wts[ss];
        }

// Parallel over design points
#pragma omp parallel for
        for (std::size_t m = 0; m < des_num; m++)
        {
            std::vector<double> xprime(dim_num, 0.0);
            std::vector<double> tmpvec(dim_num, 0.0);

            // interactions with other design points
            for (std::size_t o = 0; o < des_num; o++)
            {
                if (o != m)
                {
                    double tmptol = 0.0;
                    for (int n = 0; n < dim_num; n++)
                    {
                        tmpvec[n] = prevdes[n + m * dim_num] - prevdes[n + o * dim_num];
                        tmptol += std::pow(tmpvec[n], 2.0);
                    }
                    tmptol = std::sqrt(tmptol);
                    for (int n = 0; n < dim_num; n++)
                        xprime[n] += tmpvec[n] / (tmptol + nug * DBL_MIN);
                }
            }

            for (int n = 0; n < dim_num; n++)
                xprime[n] = xprime[n] * ((double)point_num / (double)des_num);

            // interactions with sample points
            for (std::size_t o = 0; o < point_num; o++)
            {
                double tmptol = 0.0;
                for (int n = 0; n < dim_num; n++)
                {
                    tmptol += std::pow(rnd[o][n] - prevdes[n + m * dim_num], 2.0);
                }
                tmptol = std::sqrt(tmptol);
                curconst[m] += rnd_wts[o] / (tmptol + (nug * DBL_MIN));
                for (int n = 0; n < dim_num; n++)
                    xprime[n] += rnd_wts[o] * rnd[o][n] / (tmptol + (nug * DBL_MIN));
            }

            double denom = (1.0 - (n0 / (it_num + n0))) * runconst[m] + (n0 / (it_num + n0)) * curconst[m];
            for (int n = 0; n < dim_num; n++)
                xprime[n] = ((1.0 - (n0 / (it_num + n0))) * runconst[m] * prevdes[n + m * dim_num] + (n0 / (it_num + n0)) * xprime[n]) / denom;

            // enforce bounds - simpler version
            for (int n = 0; n < dim_num; n++)
                xprime[n] = std::min(std::max(xprime[n], bd[n][0]), bd[n][1]);

            for (int n = 0; n < dim_num; n++)
            {
                des_up[n + m * dim_num] = xprime[n];
                if (std::isnan(xprime[n]))
                    nanflg = true;
            }

            runconst_up[m] = (1 - (n0 / (it_num + n0))) * runconst[m] + (n0 / (it_num + n0)) * curconst[m];
        } // end parallel for

        if (nanflg)
        {
            nug += 1.0;
            std::fill(runconst.begin(), runconst.end(), 0.0);
            std::cout << "\nNumerical instabilities encountered; resetting optimization.\n";
        }
        else
        {
            des = des_up;
            runconst = runconst_up;
        }

        it_num++;
        double maxdiff = 0.0;
        double rundiff = 0.0;
        for (std::size_t n = 0; n < des_num; n++)
        {
            rundiff = 0.0;
            for (int o = 0; o < dim_num; o++)
                rundiff += std::pow(des[o + n * dim_num] - prevdes[o + n * dim_num], 2.0);
            maxdiff = std::max(maxdiff, rundiff);
        }

        if ((maxdiff < tol) && (!nanflg))
        {
            cont = false;
            std::cout << "\nTolerance level reached.";
        }

        if ((it_num >= it_max) && (!nanflg))
        {
            cont = false;
        }
    } // end while

    std::cout << "\n";
    std::vector<std::vector<double>> retdes(des_num, std::vector<double>(dim_num, 0.0));
    for (int j = 0; j < dim_num; j++)
    {
        for (std::size_t i = 0; i < des_num; i++)
        {
            retdes[i][j] = des[i * dim_num + j];
        }
    }
    return retdes;
}
