#include "xchainer/python/cuda/hello.h"

#include "xchainer/cuda/hello.h"

#include "xchainer/python/type_caster.h"  // need to include in every compilation unit of the Python extension module

namespace xchainer {
namespace cuda {

void InitXchainerCudaHello(pybind11::module& m) { m.def("hello", &Hello); }

}  // namespace cuda
}  // namespace xchainer
