#include "xchainer/routines/normalization.h"

#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class NormalizationTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(NormalizationTest, BatchNorm) {
    using T = float;

    Shape x_shape{3, 4, 2, 1};
    Shape reduced_shape{1, 4, 2, 1};
    Scalar eps{2e-5f};
    Scalar decay{0.9f};

    // Input data were generated randomly.
    Array a = testing::BuildArray(x_shape).WithData<T>({0.6742742, 0.8028925,  0.28383577, 0.8412501,  0.8006508,  0.32548666,
                                                        0.4981232, 0.2899665,  0.8781784,  0.09848342, 0.56066823, 0.46877825,
                                                        0.5734097, 0.46068498, 0.02365979, 0.40318793, 0.61877257, 0.9073324,
                                                        0.817619,  0.9549834,  0.43688482, 0.67947686, 0.62297916, 0.36094204});
    Array gamma = testing::BuildArray(reduced_shape)
                          .WithData<T>({0.67531216, 0.38460097, 0.3139644, 0.41022405, 0.3633898, 0.07180618, 0.4424598, 0.63477284});
    Array beta = testing::BuildArray(reduced_shape)
                         .WithData<T>({0.7327423, 0.6883794, 0.11482884, 0.4891287, 0.17816886, 0.26629093, 0.3904204, 0.63719493});
    Array running_mean =
            testing::BuildArray(reduced_shape)
                    .WithData<T>({0.27891612, 0.83984816, 0.20299992, 0.3024816, 0.59901035, 0.9280579, 0.07075989, 0.31253654});
    Array running_var = testing::BuildArray(reduced_shape)
                                .WithData<T>({0.8258983, 0.35525382, 0.01103283, 0.843107, 0.09379472, 0., 0.6574457, 0.6707562});
    Array out = BatchNorm(a, gamma, beta, running_mean, running_var, eps, decay);

    // Expectations were computed using Chainer.
    Array e_out = testing::BuildArray(x_shape).WithData<T>({0.43345568, 0.9024843,   -0.27429962, 0.6594735,  0.6550931,  0.18604966,
                                                            0.5901094,  -0.19329488, 1.6671101,   0.14835835, 0.12437291, -0.0761953,
                                                            0.1049637,  0.25257912,  -0.22290352, 1.3381515,  0.0976615,  1.0142955,
                                                            0.49441338, 0.88410795,  -0.22555014, 0.360244,   0.80405533, 0.7667286});
    Array e_running_mean =
            testing::BuildArray(reduced_shape)
                    .WithData<T>({0.32339868, 0.8161536, 0.23810402, 0.34773383, 0.59947413, 0.88410705, 0.10184264, 0.31641942});
    Array e_running_var =
            testing::BuildArray(reduced_shape)
                    .WithData<T>({0.7451742, 0.33908403, 0.01705596, 0.76526403, 0.08779196, 0.00319096, 0.6016993, 0.60400796});

    testing::ExpectAllClose(e_out, out, 1e-6f, 1e-6f);
    testing::ExpectAllClose(e_running_mean, running_mean, 1e-6f, 1e-6f);
    testing::ExpectAllClose(e_running_var, running_var, 1e-6f, 1e-6f);
}

TEST_P(NormalizationTest, BatchNormWithAxis) {
    using T = float;

    Shape x_shape{3, 4, 2, 1};
    Shape reduced_shape{1, 4, 1, 1};
    Axes axis{0, 2, 3};
    Scalar eps{2e-5f};
    Scalar decay{0.8f};

    // Input data were generated randomly.
    Array a = testing::BuildArray(x_shape).WithData<T>({0.03225313, 0.8745096,  0.97541857, 0.73366016, 0.10335114, 0.89237994,
                                                        0.25917393, 0.7404295,  0.62782156, 0.27429798, 0.20574076, 0.72369,
                                                        0.3420079,  0.57100123, 0.8667755,  0.02320529, 0.32140592, 0.15076979,
                                                        0.25860628, 0.14361706, 0.82823735, 0.38278055, 0.48861042, 0.7562712});
    Array gamma = testing::BuildArray(reduced_shape).WithData<T>({0.47078794, 0.50151867, 0.50990486, 0.23072837});
    Array beta = testing::BuildArray(reduced_shape).WithData<T>({0.07768852, 0.21956936, 0.6850719, 0.15088539});
    Array running_mean = testing::BuildArray(reduced_shape).WithData<T>({0.34721586, 0.2698823, 0.8581124, 0.74137366});
    Array running_var = testing::BuildArray(reduced_shape).WithData<T>({0., 0.8622455, 0.18700261, 0.20017703});
    Array out = BatchNorm(a, gamma, beta, running_mean, running_var, eps, decay, axis);

    // Expectations were computed using Chainer.
    Array e_out = testing::BuildArray(x_shape).WithData<T>({-0.4931047,  0.88867813,  0.96136284, 0.5786838,   -0.08176702, 1.3705747,
                                                            -0.05147286, 0.31848282,  0.48396856, -0.09601258, -0.25695923, 0.562902,
                                                            0.35752136,  0.7790225,   0.41560876, -0.23286907, -0.01872858, -0.29866958,
                                                            -0.17327842, -0.35529473, 1.2525094,  0.43257034,  0.12490187,  0.33066076});
    Array e_running_mean = testing::BuildArray(reduced_shape).WithData<T>({0.35380796, 0.3172636, 0.79048187, 0.6975811});
    Array e_running_var = testing::BuildArray(reduced_shape).WithData<T>({0.01976142, 0.7138863, 0.16801749, 0.18175972});

    testing::ExpectAllClose(e_out, out, 1e-4f, 1e-4f);
    testing::ExpectAllClose(e_running_mean, running_mean, 1e-6f, 1e-6f);
    testing::ExpectAllClose(e_running_var, running_var, 1e-6f, 1e-6f);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        NormalizationTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
