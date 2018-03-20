#include "xchainer/python/array.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/backward.h"
#include "xchainer/constant.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/slice.h"

#include "xchainer/python/array_index.h"
#include "xchainer/python/common.h"
#include "xchainer/python/shape.h"
#include "xchainer/python/strides.h"

namespace xchainer {

namespace py = pybind11;

namespace {

// TODO(beam2d): The current binding has an overhead on wrapping ArrayBodyPtr by Array, which copies shared_ptr. One
// simple way to avoid this overhead is to use reinterpret_cast<Array&>(ptr). This cast is valid if ArrayBodyPtr (i.e.,
// shared_ptr) satisfies "standard layout" conditions. We can test if ArrayBodyPtr satisfies these conditions by
// std::is_standard_layout (see http://en.cppreference.com/w/cpp/types/is_standard_layout#Notes).

using ArrayBodyPtr = std::shared_ptr<internal::ArrayBody>;
using ConstArrayBodyPtr = std::shared_ptr<const internal::ArrayBody>;

Dtype NumpyDtypeToDtype(const py::dtype& npdtype) {
    switch (npdtype.kind()) {
        case 'b':
            return Dtype::kBool;
        case 'i':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kInt8;
                case 2:
                    return Dtype::kInt16;
                case 4:
                    return Dtype::kInt32;
                case 8:
                    return Dtype::kInt64;
                default:
                    break;
            }
            break;
        case 'u':
            switch (npdtype.itemsize()) {
                case 1:
                    return Dtype::kUInt8;
                default:
                    break;
            }
            break;
        case 'f':
            switch (npdtype.itemsize()) {
                case 4:
                    return Dtype::kFloat32;
                case 8:
                    return Dtype::kFloat64;
                default:
                    break;
            }
            break;
        default:
            break;
    }
    throw DtypeError("unsupported NumPy dtype");
}

Device& GetDevice(const nonstd::optional<std::string>& device_id) {
    return device_id.has_value() ? GetDefaultContext().GetDevice(device_id.value()) : GetDefaultDevice();
}

ArrayBodyPtr MakeArray(const py::tuple& shape_tup, Dtype dtype, const py::list& list, const nonstd::optional<std::string>& device_id) {
    Shape shape = internal::ToShape(shape_tup);
    auto total_size = shape.GetTotalSize();
    auto bytes = GetElementSize(dtype) * total_size;
    if (static_cast<size_t>(total_size) != list.size()) {
        throw DimensionError("Invalid data length");
    }

    // Allocate a buffer and copy data
    std::shared_ptr<void> ptr = std::make_unique<uint8_t[]>(bytes);
    VisitDtype(dtype, [&](auto pt) {
        using T = typename decltype(pt)::type;
        std::transform(list.begin(), list.end(), static_cast<T*>(ptr.get()), [](auto& item) { return py::cast<T>(item); });
    });

    return Array::FromBuffer(shape, dtype, ptr, GetDevice(device_id)).move_body();
}

ArrayBodyPtr MakeArray(py::array array, const nonstd::optional<std::string>& device_id) {
    if ((array.flags() & py::array::c_style) == 0) {
        throw DimensionError("cannot convert non-contiguous NumPy array to Array");
    }

    Dtype dtype = NumpyDtypeToDtype(array.dtype());
    py::buffer_info info = array.request();
    Shape shape(info.shape);

    // data holds the copy of py::array which in turn references the NumPy array and the buffer is therefore not released
    void* underlying_data = array.mutable_data();
    std::shared_ptr<void> data{std::make_shared<py::array>(std::move(array)), underlying_data};

    return Array::FromBuffer(shape, dtype, data, GetDevice(device_id)).move_body();
}

py::buffer_info MakeNumpyArrayFromArray(internal::ArrayBody& self) {
    // Used as a temporary accessor
    Array array{std::move(ArrayBodyPtr(&self, [](internal::ArrayBody* ptr) {
        (void)ptr;  // unused
    }))};

    return py::buffer_info(
            array.data().get(),
            array.element_bytes(),
            std::string(1, GetCharCode(array.dtype())),
            array.ndim(),
            array.shape(),
            array.strides());
}

}  // namespace

void InitXchainerArray(pybind11::module& m) {
    py::class_<internal::ArrayBody, ArrayBodyPtr>{m, "Array", py::buffer_protocol()}
            .def(py::init(py::overload_cast<const py::tuple&, Dtype, const py::list&, const nonstd::optional<std::string>&>(&MakeArray)),
                 py::arg("shape"),
                 py::arg("dtype"),
                 py::arg("data"),
                 py::arg("device") = nullptr)
            .def(py::init(py::overload_cast<py::array, const nonstd::optional<std::string>&>(&MakeArray)),
                 py::arg("data"),
                 py::arg("device") = nullptr)
            .def_buffer(&MakeNumpyArrayFromArray)
            .def("view",
                 [](const ArrayBodyPtr& self) {
                     // Duplicate the array body
                     return std::make_shared<internal::ArrayBody>(*self);
                 })
            .def("__iadd__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} += Array{rhs}).move_body(); })
            .def("__imul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} *= Array{rhs}).move_body(); })
            .def("__add__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} + Array{rhs}).move_body(); })
            .def("__mul__", [](const ArrayBodyPtr& self, const ArrayBodyPtr& rhs) { return (Array{self} * Array{rhs}).move_body(); })
            .def("__repr__", [](const ArrayBodyPtr& self) { return Array{self}.ToString(); })
            .def("__getitem__",
                 [](const ArrayBodyPtr& self, py::handle handle) {
                     return Array{self}.At(python::internal::MakeArrayIndices(handle)).move_body();
                 })
            .def("to_device", [](const ArrayBodyPtr& self, Device& device) { return Array{self}.ToDevice(device).move_body(); })
            .def("to_device",
                 [](const ArrayBodyPtr& self, const std::string& device_name) {
                     Device& device = GetDefaultContext().GetDevice({device_name});
                     return Array{self}.ToDevice(device).move_body();
                 })
            .def("to_device",
                 [](const ArrayBodyPtr& self, const std::string& backend_name, int index) {
                     Device& device = GetDefaultContext().GetDevice({backend_name, index});
                     return Array{self}.ToDevice(device).move_body();
                 })
            .def("transpose", [](const ArrayBodyPtr& self) { return Array{self}.Transpose().move_body(); })
            .def("reshape",
                 [](const ArrayBodyPtr& self, py::tuple shape) { return Array{self}.Reshape(internal::ToShape(shape)).move_body(); })
            .def("reshape",
                 [](const ArrayBodyPtr& self, const std::vector<int64_t>& shape) {
                     return Array{self}.Reshape({shape.begin(), shape.end()}).move_body();
                 })
            .def("reshape",
                 [](const ArrayBodyPtr& self, py::args args) {
                     auto shape = py::cast<std::vector<int64_t>>(args);
                     return Array{self}.Reshape({shape.begin(), shape.end()}).move_body();
                 })
            .def("copy", [](const ArrayBodyPtr& self) { return Array{self}.Copy().move_body(); })
            .def("as_constant",
                 [](const ArrayBodyPtr& self, bool copy) {
                     return Array{self}.AsConstant(copy ? CopyKind::kCopy : CopyKind::kView).move_body();
                 },
                 py::arg("copy") = false)
            .def("as_constant",
                 [](const ArrayBodyPtr& self, const std::vector<GraphId>& graph_ids, bool copy) {
                     return Array{self}.AsConstant(graph_ids, copy ? CopyKind::kCopy : CopyKind::kView).move_body();
                 },
                 py::arg().noconvert(),
                 py::arg("copy") = false)
            .def("require_grad",
                 [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.RequireGrad(graph_id).move_body(); },
                 py::arg("graph_id") = kDefaultGraphId)
            .def("is_grad_required",
                 [](const ArrayBodyPtr& self, const GraphId& graph_id) { return Array{self}.IsGradRequired(graph_id); },
                 py::arg("graph_id") = kDefaultGraphId)
            .def("get_grad",
                 [](const ArrayBodyPtr& self, const GraphId& graph_id) -> ConstArrayBodyPtr {
                     const nonstd::optional<Array>& grad = Array{self}.GetGrad(graph_id);
                     if (!grad.has_value()) {
                         return nullptr;
                     }
                     return grad->body();
                 },
                 py::arg("graph_id") = kDefaultGraphId)
            .def("set_grad",
                 [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad, const GraphId& graph_id) {
                     auto array = Array{self};
                     if (grad) {
                         array.SetGrad(Array{grad}, graph_id);
                     } else {
                         array.ClearGrad(graph_id);
                     }
                 },
                 py::arg("grad"),
                 py::arg("graph_id") = kDefaultGraphId)
            .def("backward",
                 [](const ArrayBodyPtr& self, const GraphId& graph_id, bool enable_double_backprop) {
                     Array array{self};
                     auto double_backprop = enable_double_backprop ? DoubleBackpropOption::kEnable : DoubleBackpropOption::kDisable;
                     Backward(array, graph_id, double_backprop);
                 },
                 py::arg("graph_id") = kDefaultGraphId,
                 py::arg("enable_double_backprop") = false)
            .def_property(
                    "grad",
                    [](const ArrayBodyPtr& self) -> ConstArrayBodyPtr {
                        const nonstd::optional<Array>& grad = Array{self}.GetGrad(kDefaultGraphId);
                        if (!grad.has_value()) {
                            return nullptr;
                        }
                        return grad->body();
                    },
                    [](const ArrayBodyPtr& self, const ArrayBodyPtr& grad) {
                        auto array = Array{self};
                        if (grad) {
                            array.SetGrad(Array{grad}, kDefaultGraphId);
                        } else {
                            array.ClearGrad(kDefaultGraphId);
                        }
                    })
            .def("cleargrad",
                 [](const ArrayBodyPtr& self, const GraphId& graph_id) { Array{self}.ClearGrad(graph_id); },
                 py::arg("graph_id") = kDefaultGraphId)
            .def_property_readonly(
                    "device", [](const ArrayBodyPtr& self) -> Device& { return Array{self}.device(); }, py::return_value_policy::reference)
            .def_property_readonly("dtype", [](const ArrayBodyPtr& self) { return Array{self}.dtype(); })
            .def_property_readonly("element_bytes", [](const ArrayBodyPtr& self) { return Array{self}.element_bytes(); })
            .def_property_readonly("is_contiguous", [](const ArrayBodyPtr& self) { return Array{self}.IsContiguous(); })
            .def_property_readonly("ndim", [](const ArrayBodyPtr& self) { return Array{self}.ndim(); })
            .def_property_readonly("offset", [](const ArrayBodyPtr& self) { return Array{self}.offset(); })
            .def_property_readonly("shape", [](const ArrayBodyPtr& self) { return internal::ToTuple(Array{self}.shape()); })
            .def_property_readonly("strides", [](const ArrayBodyPtr& self) { return internal::ToTuple(Array{self}.strides()); })
            .def_property_readonly("total_bytes", [](const ArrayBodyPtr& self) { return Array{self}.GetTotalBytes(); })
            .def_property_readonly("total_size", [](const ArrayBodyPtr& self) { return Array{self}.GetTotalSize(); })
            .def_property_readonly("T", [](const ArrayBodyPtr& self) { return Array{self}.Transpose().move_body(); })
            .def_property_readonly(
                    "_debug_data_memory_address",  // These methods starting with `_debug_` are stubs for testing
                    [](const ArrayBodyPtr& self) -> intptr_t {
                        const void* ptr = Array{self}.data().get();
                        return reinterpret_cast<intptr_t>(ptr);  // NOLINT: reinterpret_cast
                    })
            .def_property_readonly("_debug_flat_data", [](const ArrayBodyPtr& self) {
                py::list list;
                Array array{self};

                // Copy data into the list
                VisitDtype(array.dtype(), [&array, &list](auto pt) {
                    using T = typename decltype(pt)::type;
                    IndexableArray<const T> iarray{array};
                    Indexer<> indexer{array.shape()};

                    for (int64_t i = 0; i < indexer.total_size(); ++i) {
                        indexer.Set(i);
                        list.append(iarray[indexer]);
                    }
                });

                return list;
            });

    m.def("empty",
          [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
              return Array::Empty(internal::ToShape(shape), dtype, GetDevice(device_id)).move_body();
          },
          py::arg("shape"),
          py::arg("dtype"),
          py::arg("device") = nullptr)
            .def("full",
                 [](py::tuple shape, Scalar value, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                     return Array::Full(internal::ToShape(shape), value, dtype, GetDevice(device_id)).move_body();
                 },
                 py::arg("shape"),
                 py::arg("value"),
                 py::arg("dtype"),
                 py::arg("device") = nullptr)
            .def("full",
                 [](py::tuple shape, Scalar value, const nonstd::optional<std::string>& device_id) {
                     return Array::Full(internal::ToShape(shape), value, GetDevice(device_id)).move_body();
                 },
                 py::arg("shape"),
                 py::arg("value"),
                 py::arg("device") = nullptr)
            .def("zeros",
                 [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                     return Array::Zeros(internal::ToShape(shape), dtype, GetDevice(device_id)).move_body();
                 },
                 py::arg("shape"),
                 py::arg("dtype"),
                 py::arg("device") = nullptr)
            .def("ones",
                 [](py::tuple shape, Dtype dtype, const nonstd::optional<std::string>& device_id) {
                     return Array::Ones(internal::ToShape(shape), dtype, GetDevice(device_id)).move_body();
                 },
                 py::arg("shape"),
                 py::arg("dtype"),
                 py::arg("device") = nullptr)
            .def("empty_like",
                 [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                     return Array::EmptyLike(Array{other}, GetDevice(device_id)).move_body();
                 },
                 py::arg("other"),
                 py::arg("device") = nullptr)
            .def("full_like",
                 [](const ArrayBodyPtr& other, Scalar value, const nonstd::optional<std::string>& device_id) {
                     return Array::FullLike(Array{other}, value, GetDevice(device_id)).move_body();
                 },
                 py::arg("other"),
                 py::arg("value"),
                 py::arg("device") = nullptr)
            .def("zeros_like",
                 [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                     return Array::ZerosLike(Array{other}, GetDevice(device_id)).move_body();
                 },
                 py::arg("other"),
                 py::arg("device") = nullptr)
            .def("ones_like",
                 [](const ArrayBodyPtr& other, const nonstd::optional<std::string>& device_id) {
                     return Array::OnesLike(Array{other}, GetDevice(device_id)).move_body();
                 },
                 py::arg("other"),
                 py::arg("device") = nullptr);

    m.def("broadcast_to",
          [](const ArrayBodyPtr& self, py::tuple shape) { return Array{self}.BroadcastTo(internal::ToShape(shape)).move_body(); },
          py::arg("array"),
          py::arg("shape"));
}

}  // namespace xchainer
