/*
 * File Name: twinning.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)
 *
 * Project: InsurAutoML
 * Latest Version: 0.2.6
 * Relative Path: /InsurAutoML/ext/twinning/twinning_cpp/twinning.cpp
 * File Created: Thursday, 6th November 2025 5:56:27 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * Last Modified: Thursday, 6th November 2025 6:34:19 pm
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

#include <vector>
#include <memory>
#include <cmath>
#include "nanoflann.hpp"

/*
 * Pure C++ implementation of twinning logic.
 * This file intentionally does NOT depend on pybind11 or any Python headers.
 * The py<->C++ conversion is handled in bindings.cpp.
 */

class DF
{
private:
    const double *data_;
    std::size_t nrow_;
    std::size_t ncol_;

public:
    DF(const double *data, std::size_t nrow, std::size_t ncol) : data_(data), nrow_(nrow), ncol_(ncol) {}

    /* functions required by nanoflann */
    std::size_t kdtree_get_point_count() const { return nrow_; }

    double kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return data_[idx * ncol_ + dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX &) const { return false; }

    /* functions used while twinning */
    const double *get_row(const std::size_t idx) const { return data_ + idx * ncol_; }
    std::size_t nrow() const { return nrow_; }
    std::size_t ncol() const { return ncol_; }
};

typedef nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Adaptor<double, DF>, DF, -1, std::size_t> KDTree;

class Twinning
{
private:
    const std::size_t r_;
    const std::size_t u1_;
    const std::size_t leaf_size_;
    DF df_;

public:
    Twinning(const double *data, std::size_t nrow, std::size_t ncol, std::size_t r, std::size_t u1, std::size_t leaf_size)
        : r_(r), u1_(u1), leaf_size_(leaf_size), df_(data, nrow, ncol) {}

    std::vector<std::vector<std::size_t>> twin()
    {
        std::size_t N = df_.nrow();
        std::size_t dim = df_.ncol();

        KDTree tree(dim, df_, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_size_));

        nanoflann::KNNResultSet<double> resultSet(r_);
        std::vector<std::size_t> index(r_);
        std::vector<double> distance(r_);

        nanoflann::KNNResultSet<double> resultSet_next_u(1);
        std::size_t index_next_u;
        double distance_next_u;

        std::vector<std::vector<std::size_t>> groups;
        groups.reserve(N / r_ + 1);
        std::size_t position = u1_;

        // track removed points so we can collect the final small group
        std::vector<char> removed(N, 0);

        while (true)
        {
            resultSet.init(index.data(), distance.data());
            tree.findNeighbors(resultSet, df_.get_row(position));

            // collect full group of r_ indices
            std::vector<std::size_t> grp;
            grp.reserve(r_);
            for (std::size_t i = 0; i < r_; ++i)
            {
                grp.push_back(index[i]);
                // mark removed and remove from tree
                removed[index[i]] = 1;
                tree.removePoint(index[i]);
            }
            groups.push_back(std::move(grp));

            // find next seed position
            resultSet_next_u.init(&index_next_u, &distance_next_u);
            tree.findNeighbors(resultSet_next_u, df_.get_row(index[r_ - 1]));
            position = index_next_u;

            // if remaining points are <= r_, gather them and finish
            std::size_t removed_count = 0;
            for (char c : removed)
                if (c)
                    ++removed_count;
            if (N - removed_count <= r_)
            {
                std::vector<std::size_t> last;
                last.reserve(N - removed_count);
                for (std::size_t i = 0; i < N; ++i)
                    if (!removed[i])
                        last.push_back(i);
                if (!last.empty())
                    groups.push_back(std::move(last));
                break;
            }
        }

        return groups;
    }
};

std::vector<std::vector<std::size_t>> twin_cpp(const double *data, std::size_t nrow, std::size_t ncol, std::size_t r, std::size_t u1, std::size_t leaf_size)
{
    Twinning twinning(data, nrow, ncol, r, u1, leaf_size);
    return twinning.twin();
}
