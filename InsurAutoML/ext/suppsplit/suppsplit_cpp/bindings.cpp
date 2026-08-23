/*
 * File Name: bindings.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Mathematics Department, University of Illinois at Urbana-Champaign (UIUC)
 *
 * Project: InsurAutoML
 * Latest Version: 0.2.6
 * Relative Path: /InsurAutoML/ext/suppsplit/suppsplit_cpp/bindings.cpp
 * File Created: Thursday, 11th September 2025 2:46:54 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * Last Modified: Thursday, 6th November 2025 7:47:39 pm
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

// bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

// declare functions implemented in sp.cpp and SPlit.cpp
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
                                        bool rnd_flg);

std::vector<std::size_t> subsample(const std::vector<std::vector<double>> &data, const std::vector<std::vector<double>> &points);

// helper: convert py::array_t<double> -> std::vector<std::vector<double>>
std::vector<std::vector<double>> ndarray_to_vector2d(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info info = arr.request();
    if (info.ndim != 2)
        throw std::runtime_error("Array must be 2-D");
    size_t rows = (size_t)info.shape[0];
    size_t cols = (size_t)info.shape[1];
    double *data_ptr = static_cast<double *>(info.ptr);
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            out[i][j] = data_ptr[i * cols + j];
    return out;
}

// helper: convert py array 1-D to std::vector<double>
std::vector<double> ndarray_to_vector1d(py::array_t<double, py::array::c_style | py::array::forcecast> arr)
{
    py::buffer_info info = arr.request();
    if (info.ndim != 1)
        throw std::runtime_error("Array must be 1-D");
    size_t n = (size_t)info.shape[0];
    double *data_ptr = static_cast<double *>(info.ptr);
    return std::vector<double>(data_ptr, data_ptr + n);
}

py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<double>> &M)
{
    size_t rows = M.size();
    size_t cols = rows ? M[0].size() : 0;
    auto result = py::array_t<double>({rows, cols});
    py::buffer_info rb = result.request();
    double *rp = static_cast<double *>(rb.ptr);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            rp[i * cols + j] = M[i][j];
    return result;
}

py::array_t<size_t> vec_to_numpy_size_t(const std::vector<std::size_t> &v)
{
    size_t n = v.size();
    auto result = py::array_t<size_t>(n);
    py::buffer_info rb = result.request();
    size_t *rp = static_cast<size_t *>(rb.ptr);
    for (size_t i = 0; i < n; ++i)
        rp[i] = v[i];
    return result;
}

PYBIND11_MODULE(suppsplit_cpp, m)
{
    m.doc() = "suppsplit_cpp: C++ support points and subsample backend (std::vector + nanoflann)";

    m.def("sp_cpp", [](py::array_t<double> ini, py::array_t<double> distsamp, bool thin, py::array_t<double> bd, std::size_t point_num, int it_max, double tol, int num_proc, double n0, py::array_t<double> wts, bool rnd_flg)
          {

        auto Ini = ndarray_to_vector2d(ini);
        auto Distsamp = ndarray_to_vector2d(distsamp);
        auto Bd = ndarray_to_vector2d(bd);
        auto Wts = ndarray_to_vector1d(wts);

        std::size_t des_num = Ini.size();
        int dim_num = des_num ? Ini[0].size() : 0;

        // Redirect std::cout to Python's sys.stdout
        py::scoped_ostream_redirect stream_cout(
            std::cout,
            py::module_::import("sys").attr("stdout")
        );

        auto out = sp_cpp(des_num, dim_num, Ini, Distsamp, thin, Bd, point_num, it_max, tol, num_proc, n0, Wts, rnd_flg);
        return vector2d_to_numpy(out); }, py::arg("ini"), py::arg("distsamp"), py::arg("thin"), py::arg("bd"), py::arg("point_num"), py::arg("it_max"), py::arg("tol"), py::arg("num_proc"), py::arg("n0"), py::arg("wts"), py::arg("rnd_flg"));

    m.def("subsample", [](py::array_t<double> data, py::array_t<double> points)
          {
        auto Data = ndarray_to_vector2d(data);
        auto Points = ndarray_to_vector2d(points);
        std::vector<std::size_t> idx = subsample(Data, Points);
        return vec_to_numpy_size_t(idx); }, py::arg("data"), py::arg("points"));
}
