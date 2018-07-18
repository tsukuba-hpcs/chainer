#include "xchainer/graph.h"

#include <ostream>

namespace xchainer {

std::ostream& operator<<(std::ostream& os, const GraphId& graph_id) {
    // TODO(niboshi): Implement graph name lookup
    return os << graph_id.sub_id();
}

}  // namespace xchainer
