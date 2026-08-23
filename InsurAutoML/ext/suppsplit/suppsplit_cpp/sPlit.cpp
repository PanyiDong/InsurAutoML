/*
 * File Name: sPlit.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)
 *
 * Project: src
 * Latest Version: <<projectversion>>
 * Relative Path: /sPlit.cpp
 * File Created: Thursday, 11th September 2025 2:45:03 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * Last Modified: Monday, 15th September 2025 2:10:58 pm
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

// SPlit.cpp
// Plain C++ version of original SPlit.cpp using nanoflann + Armadillo
#include <memory>
#include <vector>
#include <iostream>
#include <vector>
#include "nanoflann.hpp"

// Data adaptor for nanoflann backed by arma::mat (rows = points, cols = dims)
class DF
{

private:
    const std::vector<std::vector<double>> *mat_; // pointer to (N x dim) matrix

public:
    DF() : mat_(nullptr) {}

    void import_data(const std::vector<std::vector<double>> &df)
    {
        mat_ = &df;
    }

    inline std::size_t kdtree_get_point_count() const
    {
        return mat_->size();
    }

    inline double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return (*mat_)[idx][dim];
    }

    // returns pointer to the row (point) data (double pointer)
    const double *get_row(const std::size_t idx) const
    {
        return mat_->at(idx).data();
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }
};

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Adaptor<double, DF>,
    DF,
    -1,
    std::size_t>
    kdTree;

class KDTree
{

private:
    const std::size_t dim_;
    const std::size_t N_;
    const std::size_t n_;
    DF data_;
    DF sp_;

public:
    KDTree(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &sp)
        : dim_(data[0].size()), N_(data.size()), n_(sp.size())
    {
        if (sp[0].size() != dim_)
        {
            std::cerr << "Dimensions do not match.\n";
            return;
        }
        data_.import_data(data);
        sp_.import_data(sp);
    }

    std::vector<std::size_t> subsample_indices_sequential()
    {
        kdTree tree(dim_, data_, nanoflann::KDTreeSingleIndexAdaptorParams(8));
        nanoflann::KNNResultSet<double> resultSet(1);
        std::size_t index;
        double distance;

        std::vector<std::size_t> indices;
        indices.reserve(n_);

        for (std::size_t i = 0; i < n_; i++)
        {
            resultSet.init(&index, &distance);
            tree.findNeighbors(resultSet, sp_.get_row(i), nanoflann::SearchParameters());
            indices.push_back(index + 1);
            tree.removePoint(index);
        }

        return indices;
    }
};

std::vector<std::size_t> subsample(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &points)
{
    KDTree kdt(data, points);
    return kdt.subsample_indices_sequential();
}
