#include "xchainer/indexer.h"

#include <cstdint>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace xchainer {
namespace {

template <int8_t kNdim>
std::string ToString(const Indexer<kNdim>& indexer) {
    std::ostringstream os;
    os << indexer;
    return os.str();
}

TEST(IndexerTest, Rank0) {
    Indexer<> indexer({});
    EXPECT_EQ(0, indexer.ndim());
    EXPECT_EQ(1, indexer.total_size());
    EXPECT_EQ("Indexer(shape=())", ToString(indexer));
}

TEST(IndexerTest, Rank1) {
    Indexer<> indexer({3});
    EXPECT_EQ(1, indexer.ndim());
    EXPECT_EQ(3, indexer.total_size());
    EXPECT_EQ(3, indexer.shape()[0]);
    EXPECT_EQ("Indexer(shape=(3,))", ToString(indexer));
}

TEST(IndexerTest, Rank3) {
    Indexer<> indexer({2, 3, 4});
    EXPECT_EQ(3, indexer.ndim());
    EXPECT_EQ(2 * 3 * 4, indexer.total_size());

    EXPECT_EQ(2, indexer.shape()[0]);
    EXPECT_EQ(3, indexer.shape()[1]);
    EXPECT_EQ(4, indexer.shape()[2]);

    EXPECT_EQ("Indexer(shape=(2, 3, 4))", ToString(indexer));
}

}  // namespace
}  // namespace xchainer
