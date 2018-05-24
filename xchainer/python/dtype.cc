#include "xchainer/python/dtype.h"

#include <string>

#include "xchainer/dtype.h"
#include "xchainer/scalar.h"

#include "xchainer/python/common.h"

namespace xchainer {
namespace python {
namespace internal {

namespace py = pybind11;  // standard convention

Dtype GetDtypeFromString(const std::string& name) {
    // From Python type names
    if (name == "bool") {
        return Dtype::kBool;
    }
    if (name == "int") {
        // NumPy returns the dtype corresponding to C long.
        // xChainer does not follow that.
        return Dtype::kInt64;
    }
    if (name == "float") {
        return Dtype::kFloat64;
    }

    // Alias specific to Python binding
    if (name == "bool_") {
        return Dtype::kBool;
    }

    // From Dtype names
    return xchainer::GetDtype(name);
}

Dtype GetDtypeFromNumpyDtype(const py::dtype& npdtype) {
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
    throw DtypeError{"unsupported NumPy dtype"};
}

Dtype GetDtype(py::handle handle) {
    // From xchainer::Dtype
    if (py::isinstance<Dtype>(handle)) {
        return py::cast<Dtype>(handle);
    }

    // From Python types
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyBool_Type)) {  // NOLINT: reinterpret_cast
        return Dtype::kBool;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyLong_Type)) {  // NOLINT: reinterpret_cast
        return Dtype::kInt64;
    }
    if (handle.ptr() == reinterpret_cast<PyObject*>(&PyFloat_Type)) {  // NOLINT: reinterpret_cast
        return Dtype::kFloat64;
    }

    // From NumPy dtypes
    if (py::isinstance<py::dtype>(handle)) {
        return GetDtypeFromNumpyDtype(py::cast<py::dtype>(handle));
    }

    // From string
    if (py::isinstance<py::str>(handle)) {
        return GetDtypeFromString(py::cast<std::string>(handle));
    }

    throw py::type_error{"Dtype not understood: " + py::cast<std::string>(py::repr(handle))};
}

namespace {

py::str GetKindStr(Dtype dtype) { return py::dtype{GetDtypeName(dtype)}.attr("kind"); }

}  // namespace

void InitXchainerDtype(pybind11::module& m) {
    py::enum_<Dtype> e{m, "dtype"};
    for (Dtype dtype : GetAllDtypes()) {
        e.value(dtype == Dtype::kBool ? "bool_" : GetDtypeName(dtype), dtype);
    }
    e.export_values();
    e.def(py::init(&internal::GetDtype));
    e.def_property_readonly("char", [](Dtype dtype) { return std::string(1, GetCharCode(dtype)); });
    e.def_property_readonly("itemsize", &GetItemSize);
    e.def_property_readonly("name", &GetDtypeName);
    e.def_property_readonly("kind", &GetKindStr);
    e.def_property_readonly("num", [](Dtype self) -> py::object { return py::dtype{GetDtypeName(self)}.attr("num"); });
    e.def_property_readonly("byteorder", [](Dtype self) -> py::str {
        if (GetItemSize(self) == 1) {
            return "|";  // "not applicable"
        }
        return "=";  // "native"
    });
    e.def_property_readonly("str", [](Dtype self) -> py::str {
        std::string s{};
        int64_t itemsize = GetItemSize(self);
        if (itemsize == 1) {
            s += "|";  // "not applicable"
        } else {
            static const uint16_t kNum16 = 0xff00U;
            if (reinterpret_cast<const uint8_t*>(&kNum16)[0] == 0x00U) {  // NOLINT: reinterpret_cast
                s += "<";  // little endian
            } else {
                s += ">";  // big endian
            }
        }
        s += py::cast<std::string>(GetKindStr(self));
        s += std::to_string(itemsize);
        return s;
    });
    e.def("__eq__", [](Dtype self, py::handle other) {
        (void)self;  // unused
        (void)other;  // unused
        return false;
    });
    e.def("__ne__", [](Dtype self, py::handle other) {
        (void)self;  // unused
        (void)other;  // unused
        return true;
    });
    // TODO(sonots): NumPy returns scalar object such as numpy.float32 by `type` and `type()` calls its constructor. Align with it.
    e.def("type", [](Dtype self, bool value) { return Scalar{value, self}; });
    e.def("type", [](Dtype self, int64_t value) { return Scalar{value, self}; });
    e.def("type", [](Dtype self, double value) { return Scalar{value, self}; });
}

}  // namespace internal
}  // namespace python
}  // namespace xchainer
