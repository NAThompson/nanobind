/*
    nanobind/vectorize.h: Apply a scalar function element-wise to arrays

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include <nanobind/ndarray.h>
#include <algorithm>
#include <tuple>
#include <vector>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

// One argument of a vectorized call: a CPU array or a (broadcast) scalar
template <typename T> struct vec_arg {
    ndarray<const T, device::cpu> arr;
    T scalar{};
    size_t nd = 0;

    explicit vec_arg(handle h) {
        if (try_cast(h, arr)) // note: 'arr' is invalid if this fails
            nd = arr.ndim();
        else
            scalar = cast<T>(h);
    }

    // Extent / stride of the k-th axis counted from the right (NumPy-style)
    size_t size(size_t k) const { return k < nd ? arr.shape(nd - 1 - k) : 1; }
    int64_t stride(size_t k) const { return size(k) == 1 ? 0 : arr.stride(nd - 1 - k); }
    const T &operator[](int64_t i) const { return arr.is_valid() ? arr.data()[i] : scalar; }
};

template <size_t> using vec_obj = object;

template <typename R, typename... A, size_t... I>
auto vectorize_impl(R (*f)(A...), std::index_sequence<I...>) {
    return [f](vec_obj<I>... args) {
        std::tuple<vec_arg<std::decay_t<A>>...> v(args...);

        // Broadcast the input shapes
        size_t nd = std::max({ std::get<I>(v).nd... }), total = 1;
        std::vector<size_t> shape(nd, 1);
        for (size_t k = 0; k < nd; ++k) {
            size_t &s = shape[nd - 1 - k];
            auto bcast = [&](size_t n) {
                if (n != 1 && s != 1 && n != s)
                    throw std::invalid_argument("nb::vectorize(): shape mismatch");
                if (n != 1)
                    s = n;
            };
            (bcast(std::get<I>(v).size(k)), ...);
            if (s && total > SIZE_MAX / s)
                throw std::bad_alloc();
            total *= s;
        }

        // Loop over the output in C order, advancing each input by its strides
        R *out = new R[total];
        capsule owner(out, [](void *p) noexcept { delete[] (R *) p; });
        std::vector<size_t> idx(nd);
        int64_t off[sizeof...(A)] = { };
        for (size_t i = 0; i < total; ++i) {
            out[i] = f(std::get<I>(v)[off[I]]...);
            for (size_t k = 0; k < nd; ++k) {
                ((off[I] += std::get<I>(v).stride(k)), ...);
                if (++idx[k] < shape[nd - 1 - k])
                    break;
                ((off[I] -= (int64_t) idx[k] * std::get<I>(v).stride(k)), ...);
                idx[k] = 0;
            }
        }
        return ndarray<numpy, R>(out, nd, shape.data(), owner);
    };
}

NAMESPACE_END(detail)

/// Wrap a scalar function ``R f(A...)`` so that it broadcasts over arrays
template <typename R, typename... A> auto vectorize(R (*f)(A...)) {
    static_assert(sizeof...(A) > 0 && std::is_arithmetic_v<R>);
    return detail::vectorize_impl(f, std::index_sequence_for<A...>());
}

NAMESPACE_END(NB_NAMESPACE)
