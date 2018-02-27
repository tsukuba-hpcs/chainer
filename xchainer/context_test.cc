#include "xchainer/context.h"

#include <future>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace {

TEST(ContextTest, Ctor) { EXPECT_NO_THROW(Context()); }

TEST(Context, GetBackend) {
    Context ctx;
    Backend& backend = ctx.GetBackend("native");
    EXPECT_EQ(&backend, &ctx.GetBackend("native"));
}

TEST(Context, GetDevice) {
    Context ctx;
    Device& device = ctx.GetDevice({"native", 0});
    EXPECT_EQ(&device, &ctx.GetDevice({"native:0"}));
}

TEST(ContextTest, DefaultContext) {
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(nullptr);
    ASSERT_THROW(GetDefaultContext(), XchainerError);

    Context ctx;
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetDefaultContext());

    Context global_ctx;
    SetGlobalDefaultContext(&global_ctx);
    SetDefaultContext(nullptr);
    ASSERT_EQ(&global_ctx, &GetDefaultContext());

    SetGlobalDefaultContext(&global_ctx);
    SetDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetDefaultContext());
}

TEST(ContextTest, GlobalDefaultContext) {
    SetGlobalDefaultContext(nullptr);
    ASSERT_THROW(GetGlobalDefaultContext(), XchainerError);

    Context ctx;
    SetGlobalDefaultContext(&ctx);
    ASSERT_EQ(&ctx, &GetGlobalDefaultContext());
}

TEST(ContextTest, ThreadLocal) {
    Context ctx;
    SetDefaultContext(&ctx);

    Context ctx2;
    auto future = std::async(std::launch::async, [&ctx2] {
        SetDefaultContext(&ctx2);
        return &GetDefaultContext();
    });
    ASSERT_NE(&GetDefaultContext(), future.get());
}

TEST(ContextTest, ContextScopeCtor) {
    SetGlobalDefaultContext(nullptr);
    SetDefaultContext(nullptr);
    Context ctx1;
    {
        // ContextScope should work even if default context is not set
        ContextScope scope(ctx1);
        EXPECT_EQ(&ctx1, &GetDefaultContext());
    }
    ASSERT_THROW(GetDefaultContext(), XchainerError);
    SetDefaultContext(&ctx1);
    {
        Context ctx2;
        ContextScope scope(ctx2);
        EXPECT_EQ(&ctx2, &GetDefaultContext());
    }
    ASSERT_EQ(&ctx1, &GetDefaultContext());
    {
        ContextScope scope;
        EXPECT_EQ(&ctx1, &GetDefaultContext());
        Context ctx2;
        SetDefaultContext(&ctx2);
    }
    ASSERT_EQ(&ctx1, &GetDefaultContext());
    Context ctx2;
    {
        ContextScope scope(ctx2);
        scope.Exit();
        EXPECT_EQ(&ctx1, &GetDefaultContext());
        SetDefaultContext(&ctx2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(&ctx2, &GetDefaultContext());
}

}  // namespace
}  // namespace xchainer
