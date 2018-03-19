#include <pybind11/pybind11.h>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/python/array.h"
#include "xchainer/python/array_index.h"
#include "xchainer/python/backend.h"
#include "xchainer/python/backward.h"
#include "xchainer/python/check_backward.h"
#include "xchainer/python/common.h"
#include "xchainer/python/context.h"
#include "xchainer/python/device.h"
#include "xchainer/python/dtype.h"
#include "xchainer/python/error.h"
#include "xchainer/python/scalar.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

namespace xchainer {
namespace {

void InitXchainerModule(pybind11::module& m) {
    m.doc() = "xChainer";
    m.attr("__name__") = "xchainer";  // Show each member as "xchainer.*" instead of "xchainer.core.*"

    m.attr("DEFAULT_GRAPH_ID") = kDefaultGraphId;
}

}  // namespace
}  // namespace xchainer

PYBIND11_MODULE(_core, m) {  // NOLINT
    xchainer::InitXchainerModule(m);
    xchainer::InitXchainerContext(m);
    xchainer::InitXchainerBackend(m);
    xchainer::InitXchainerDevice(m);
    xchainer::InitXchainerDtype(m);
    xchainer::InitXchainerError(m);
    xchainer::InitXchainerScalar(m);
    xchainer::InitXchainerShape(m);
    xchainer::InitXchainerStrides(m);
    xchainer::InitXchainerArrayIndex(m);
    xchainer::InitXchainerArray(m);
    xchainer::InitXchainerBackward(m);
    xchainer::InitXchainerCheckBackward(m);
}
