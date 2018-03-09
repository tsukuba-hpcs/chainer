#include <iostream>

#include <gtest/gtest.h>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/testing/util.h"

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int status = RUN_ALL_TESTS();

    if (xchainer::testing::GetSkippedNativeTestCount() > 0) {
        xchainer::Context ctx;
        xchainer::Backend& backend = ctx.GetBackend("native");
        std::cout << "[  SKIPPED ] " << xchainer::testing::GetSkippedNativeTestCount() << " NATIVE tests requiring devices more than "
                  << xchainer::testing::GetDeviceLimit(backend) << "." << std::endl;
    }
#ifdef XCHAINER_ENABLE_CUDA
    if (xchainer::testing::GetSkippedCudaTestCount() > 0) {
        xchainer::Context ctx;
        xchainer::Backend& backend = ctx.GetBackend("cuda");
        std::cout << "[  SKIPPED ] " << xchainer::testing::GetSkippedCudaTestCount() << " CUDA tests requiring devices more than "
                  << xchainer::testing::GetDeviceLimit(backend) << "." << std::endl;
    }
#endif  // XCHAINER_ENABLE_CUDA

    return status;
}
