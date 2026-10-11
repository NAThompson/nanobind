#include <nanobind/vectorize.h>
#include <cmath>

namespace nb = nanobind;

static double add(double a, double b) { return a + b; }
static double hyp(double a, double b) { return std::sqrt(a * a + b * b); }
static int64_t mul3(int64_t a, int64_t b, int64_t c) { return a * b * c; }

NB_MODULE(test_vectorize_ext, m) {
    m.def("square", nb::vectorize(+[](double x) { return x * x; }));
    m.def("add", nb::vectorize(&add));
    m.def("hyp", nb::vectorize(&hyp));
    m.def("mul3", nb::vectorize(&mul3));
}
