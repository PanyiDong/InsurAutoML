/*
 * File Name: bindings.cpp
 * Author: Panyi Dong
 * GitHub: https://github.com/PanyiDong/
 * Actuarial and Risk Management Sciences, University of Illinois at Urbana-Champaign (UIUC)
 *
 * Project: InsurAutoML
 * Latest Version: 0.2.6
 * Relative Path: /InsurAutoML/ext/twinning/twinning_cpp/bindings.cpp
 * File Created: Thursday, 6th November 2025 6:20:09 pm
 * Author: Panyi Dong (panyid2@illinois.edu)
 *
 * -----
 * Last Modified: Thursday, 6th November 2025 6:30:25 pm
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

// declare functions implemented in twinning.cpp (pure C++ interface)
std::vector<std::vector<std::size_t>> twin_cpp(const double *data, std::size_t nrow, std::size_t ncol, std::size_t r, std::size_t u1, std::size_t leaf_size);

py::array_t<std::size_t> vec_to_numpy_size_t(const std::vector<std::size_t> &v)
{
    size_t n = v.size();
    auto result = py::array_t<std::size_t>(n);
    py::buffer_info rb = result.request();
    std::size_t *rp = static_cast<std::size_t *>(rb.ptr);
    for (size_t i = 0; i < n; ++i)
        rp[i] = v[i];
    return result;
}

py::list vec2d_to_pylist(const std::vector<std::vector<std::size_t>> &M)
{
    py::list out;
    for (const auto &row : M)
    {
        auto arr = py::array_t<std::size_t>(row.size());
        py::buffer_info rb = arr.request();
        std::size_t *rp = static_cast<std::size_t *>(rb.ptr);
        for (size_t i = 0; i < row.size(); ++i)
            rp[i] = row[i];
        out.append(arr);
    }
    return out;
}

PYBIND11_MODULE(twinning_cpp, m)
{
    m.doc() = "twinning_cpp: C++ twinning extension";

    m.def("twin_cpp", [](py::array_t<double> data, std::size_t r, std::size_t u1, std::size_t leaf_size)
          {
              // redirect stdout
              py::scoped_ostream_redirect stream_cout(
                  std::cout, py::module_::import("sys").attr("stdout"));

              // Ensure input is a 2-D array
              py::buffer_info buf = data.request();
              if (buf.ndim != 2)
                  throw std::runtime_error("twin_cpp requires a 2-D numpy array");

              std::size_t nrow = static_cast<std::size_t>(buf.shape[0]);
              std::size_t ncol = static_cast<std::size_t>(buf.shape[1]);

              // Copy data to a contiguous std::vector<double> to ensure lifetime and layout
              const double *src = static_cast<const double *>(buf.ptr);
              std::vector<double> storage(src, src + (nrow * ncol));

              std::vector<std::vector<std::size_t>> out = twin_cpp(storage.data(), nrow, ncol, r, u1, leaf_size);
              return vec2d_to_pylist(out); }, py::arg("data"), py::arg("r"), py::arg("u1"), py::arg("leaf_size"));
}