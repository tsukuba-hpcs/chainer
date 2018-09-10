#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <future>
#include <type_traits>
#include <vector>

#include "chainerx/array_body_leak_detection.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace testing {
namespace threading_detail {

// Called by thread dispatcher.
template <typename Func>
auto CallFunc(const Func& func, size_t thread_index) -> decltype(func(size_t{})) {
    return func(thread_index);
}

// Called by thread dispatcher.
template <typename Func>
auto CallFunc(const Func& func, size_t /*thread_index*/) -> decltype(func()) {
    return func();
}

}  // namespace threading_detail

template <
        typename Func,
        typename ResultType = decltype(threading_detail::CallFunc(std::declval<Func>(), size_t{})),
        std::enable_if_t<!std::is_void<ResultType>::value, std::nullptr_t> = nullptr>
std::vector<ResultType> RunThreads(size_t thread_count, const Func& func) {
    std::mutex mutex;
    std::condition_variable cv_all_ready;

    size_t wait_count = thread_count;

    auto thread_proc = [&mutex, &cv_all_ready, &wait_count, &func](size_t thread_index) mutable -> ResultType {
        {
            std::unique_lock<std::mutex> lock{mutex};

            --wait_count;

            if (wait_count == 0) {
                // If this thread is the last one to be ready, wake up all other threads.
                cv_all_ready.notify_all();
            } else {
                // Otherwise, wait for all other threads to be ready.
                cv_all_ready.wait(lock);
            }
        }

        return threading_detail::CallFunc(func, thread_index);
    };

    // Launch threads
    std::vector<std::future<ResultType>> futures;
    futures.reserve(thread_count);

    for (size_t i = 0; i < thread_count; ++i) {
        futures.emplace_back(std::async(std::launch::async, thread_proc, i));
    }

    // Retrieve results
    std::vector<ResultType> results;
    results.reserve(thread_count);
    std::transform(
            futures.begin(), futures.end(), std::back_inserter(results), [](std::future<ResultType>& future) { return future.get(); });

    return results;
}

template <
        typename Func,
        typename ResultType = decltype(threading_detail::CallFunc(std::declval<Func>(), size_t{})),
        std::enable_if_t<std::is_void<ResultType>::value, std::nullptr_t> = nullptr>
void RunThreads(size_t thread_count, const Func& func) {
    // Call overload by wrapping the given function that returns void with a lambda that returns a nullptr.
    RunThreads(thread_count, [&func](size_t thread_index) {
        threading_detail::CallFunc(func, thread_index);
        return nullptr;
    });
}

// TODO(sonots): Reconsider the function name.
// TODO(sonots): Do single-shot and multi-threads tests in seperated test-cases.
// TODO(sonots): Make it possible to use different contexts and/or devices in different threads.
template <typename Func>
inline void RunTestWithThreads(const Func& func, size_t thread_count = 2) {
    // Run single-shot
    threading_detail::CallFunc(func, 0);

    // Run in multi-threads
    if (thread_count > 0) {
        Context& context = chainerx::GetDefaultContext();
        Device& device = chainerx::GetDefaultDevice();
        RunThreads(thread_count, [&context, &device, &func](size_t thread_index) {
            chainerx::SetDefaultContext(&context);
            chainerx::SetDefaultDevice(&device);
            threading_detail::CallFunc(func, thread_index);
        });
    }
}

#define CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name) test_case_name##_##test_name##_Dummy

#define CHAINERX_TEST_P_(test_case_name, test_name, parent_class)                                                                     \
    class CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name) : public parent_class {                                          \
    public:                                                                                                                           \
        CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)() {}                                                               \
                                                                                                                                      \
    private:                                                                                                                          \
        virtual void TestBody();                                                                                                      \
        static int AddToRegistry() {                                                                                                  \
            ::testing::UnitTest::GetInstance()                                                                                        \
                    ->parameterized_test_registry()                                                                                   \
                    .GetTestCasePatternHolder<test_case_name>(#test_case_name, ::testing::internal::CodeLocation(__FILE__, __LINE__)) \
                    ->AddTestPattern(                                                                                                 \
                            #test_case_name,                                                                                          \
                            #test_name,                                                                                               \
                            new ::testing::internal::TestMetaFactory<CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)>());  \
            return 0;                                                                                                                 \
        }                                                                                                                             \
        static int registering_dummy_ GTEST_ATTRIBUTE_UNUSED_;                                                                        \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name));                                  \
    };                                                                                                                                \
    int CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)::registering_dummy_ =                                              \
            CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)::AddToRegistry();                                              \
    void CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)::TestBody()

// Thread safety test macros that replaces TEST, TEST_F, TEST_P.
// The body of the test should include a call to Run(func) where func is a lambda function that optionally accepts a size_t thread index as
// argument and defines the logic that is executed on multiple threads.
//
// Example:
//
// TEST_THREAD_SAFE(MyTestCase, MyTest) {
//     Context ctx{};
//
//     Run([&](size_t thread_index) {
//         // Define logic that is to be executed on multiple threads.
//         BackpropId backprop_id = ctx.MakeBackpropId(std::to_string(thread_index));
//         EXPECT_EQ(std::to_string(thread_index), ctx.GetBackpropName(backprop_id));
//     });
// }
// TODO(hvy): Define macros TEST_THREAD_SAFE and TEST_THREAD_SAFE_P.
#define TEST_THREAD_SAFE_P(test_case_name, test_name)                                                                        \
    class CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name) : public test_case_name {                               \
    protected:                                                                                                               \
        void RunThreadSafeTestBodyWithLeakDetection(size_t thread_count) {                                                   \
            thread_count_ = thread_count;                                                                                    \
                                                                                                                             \
            internal::ArrayBodyLeakTracker tracker{};                                                                        \
            {                                                                                                                \
                internal::ArrayBodyLeakDetectionScope scope{tracker};                                                        \
                ThreadSafeTestBody();                                                                                        \
            }                                                                                                                \
            CheckAllArrayBodiesFreed(tracker);                                                                               \
        }                                                                                                                    \
                                                                                                                             \
        bool is_run_called() { return is_run_called_; }                                                                      \
                                                                                                                             \
    private:                                                                                                                 \
        void ThreadSafeTestBody();                                                                                           \
                                                                                                                             \
        template <typename Func>                                                                                             \
        void Run(const Func& func) {                                                                                         \
            if (thread_count_ > 1) {                                                                                         \
                testing::RunThreads(thread_count_, func);                                                                    \
            } else {                                                                                                         \
                testing::threading_detail::CallFunc(func, 0);                                                                \
            }                                                                                                                \
            is_run_called_ = true;                                                                                           \
        }                                                                                                                    \
                                                                                                                             \
        size_t thread_count_{0};                                                                                             \
        bool is_run_called_{false};                                                                                          \
    };                                                                                                                       \
                                                                                                                             \
    CHAINERX_TEST_P_(test_case_name, test_name##_SingleThread, CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)) { \
        RunThreadSafeTestBodyWithLeakDetection(1);                                                                           \
        CHAINERX_ASSERT(is_run_called());                                                                                    \
    }                                                                                                                        \
                                                                                                                             \
    CHAINERX_TEST_P_(test_case_name, test_name##_MultiThread, CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)) {  \
        RunThreadSafeTestBodyWithLeakDetection(2);                                                                           \
        CHAINERX_ASSERT(is_run_called());                                                                                    \
    }                                                                                                                        \
                                                                                                                             \
    void CHAINERX_TEST_DUMMY_CLASS_NAME_(test_case_name, test_name)::ThreadSafeTestBody()

}  // namespace testing
}  // namespace chainerx
