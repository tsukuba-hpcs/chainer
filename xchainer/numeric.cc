#include "xchainer/numeric.h"

#include <cmath>

#include "xchainer/array.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {

bool AllClose(const Array& a, const Array& b, double rtol, double atol) {
    if (a.shape() != b.shape()) {
        throw DimensionError("cannot compare Arrays of different shapes");
    }
    if (a.dtype() != b.dtype()) {
        throw DtypeError("cannot compare Arrays of different Dtypes");
    }

    a.device().Synchronize();
    b.device().Synchronize();

    return VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> a_iarray{a};
        IndexableArray<const T> b_iarray{b};
        Indexer<> indexer{a.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            if (std::abs(a_iarray[indexer] - b_iarray[indexer]) > atol + rtol * std::abs(b_iarray[indexer])) {
                return false;
            }
        }
        return true;
    });
}

}  // namespace xchainer
