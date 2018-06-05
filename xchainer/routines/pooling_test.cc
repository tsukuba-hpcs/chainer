#include "xchainer/routines/pooling.h"

#include <algorithm>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/device_id.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class PoolingTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(PoolingTest, MaxPooling) {
    if (GetParam() == "cuda") {
        // TODO(hvy): Test CUDA when implemented.
        return;
    }
    using T = float;

    int64_t batch_size = 3;
    int64_t channels = 4;
    Shape in_dims{4, 4};
    StackVector<int64_t, kMaxNdim> kernel_size{3, 2};
    StackVector<int64_t, kMaxNdim> stride{2, 1};
    StackVector<int64_t, kMaxNdim> pad{1, 0};
    Shape out_dims{3, 3};

    Shape x_shape{batch_size, channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape out_shape{batch_size, channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape)
                      .WithData<T>(
                              {0.4609564,  0.9677815,  0.08834644, 0.32941917, 0.8292747,  0.11845724, 0.1369321,  0.6074731,  0.78542763,
                               0.97462624, 0.7587014,  0.27017736, 0.7416107,  0.25967184, 0.34700692, 0.80568856, 0.7809916,  0.42632404,
                               0.00986403, 0.96131665, 0.3818825,  0.8825699,  0.32384965, 0.41834843, 0.04444144, 0.5976533,  0.13136163,
                               0.99313796, 0.20961371, 0.72154176, 0.8705839,  0.8913622,  0.10679037, 0.13024911, 0.29701585, 0.9328509,
                               0.29156446, 0.56731504, 0.08864314, 0.40091372, 0.77066797, 0.7060211,  0.09676889, 0.09575618, 0.8108767,
                               0.04177126, 0.04649203, 0.17163254, 0.49008262, 0.6410025,  0.8935332,  0.06818897, 0.49048236, 0.6969235,
                               0.03973121, 0.99195975, 0.92212546, 0.2568234,  0.74829006, 0.49562132, 0.73161113, 0.3562581,  0.09955528,
                               0.42987233, 0.07259833, 0.02391367, 0.6531855,  0.2526743,  0.96925163, 0.99820083, 0.35440677, 0.45040762,
                               0.8124794,  0.6873842,  0.3394505,  0.99982584, 0.37112013, 0.9059909,  0.12874502, 0.10643691, 0.21043272,
                               0.22249077, 0.2507038,  0.38035834, 0.76311666, 0.58646417, 0.73357195, 0.70151967, 0.82812095, 0.45903125,
                               0.854887,   0.9307932,  0.5138541,  0.00605829, 0.88109905, 0.05902579, 0.93474656, 0.516853,   0.80964804,
                               0.6165152,  0.8065471,  0.4231297,  0.8462578,  0.12397768, 0.96989137, 0.13212852, 0.39432606, 0.8906301,
                               0.54361,    0.05775663, 0.96815336, 0.44516703, 0.6066227,  0.10383689, 0.795366,   0.06057209, 0.8556079,
                               0.32239342, 0.9142884,  0.52067345, 0.33631396, 0.337069,   0.98927075, 0.45864356, 0.05180012, 0.6295072,
                               0.63463855, 0.99933624, 0.9241264,  0.2909103,  0.12271244, 0.43939343, 0.98111194, 0.82608557, 0.6107712,
                               0.08100884, 0.6419318,  0.80480385, 0.24884045, 0.6263302,  0.40993217, 0.6449191,  0.7088349,  0.02296176,
                               0.70677763, 0.7166788,  0.2855127,  0.39801753, 0.8171236,  0.23696144, 0.4529571,  0.5830564,  0.41618168,
                               0.6569938,  0.73607063, 0.55866545, 0.10323327, 0.10768154, 0.9575846,  0.81976444, 0.6253338,  0.1517685,
                               0.1641238,  0.94771904, 0.86659664, 0.0256371,  0.1406688,  0.107798,   0.2999732,  0.7015409,  0.7981461,
                               0.09489103, 0.8165871,  0.8357075,  0.09764841, 0.05153274, 0.8971699,  0.9327884,  0.32184523, 0.15035488,
                               0.29527086, 0.34706247, 0.08613685, 0.22496991, 0.28078404, 0.17121029, 0.4556634,  0.5025214,  0.7903231,
                               0.87756634, 0.3690981,  0.6356852})
                      .WithPadding(1);  // Computed with Chainer.

    Array out = MaxPooling(x, kernel_size, stride, pad);

    Array e_out = testing::BuildArray(out_shape).WithData<T>(
            {0.9677815,  0.9677815,  0.6074731,  0.97462624, 0.97462624, 0.80568856, 0.7416107,  0.34700692, 0.80568856, 0.8825699,
             0.8825699,  0.96131665, 0.8825699,  0.8825699,  0.99313796, 0.72154176, 0.8705839,  0.8913622,  0.56731504, 0.56731504,
             0.9328509,  0.8108767,  0.7060211,  0.40091372, 0.8108767,  0.04649203, 0.17163254, 0.6969235,  0.8935332,  0.99195975,
             0.92212546, 0.74829006, 0.99195975, 0.73161113, 0.3562581,  0.42987233, 0.99820083, 0.99820083, 0.6531855,  0.99820083,
             0.99820083, 0.99982584, 0.9059909,  0.9059909,  0.12874502, 0.76311666, 0.73357195, 0.73357195, 0.82812095, 0.88109905,
             0.9307932,  0.5138541,  0.88109905, 0.88109905, 0.93474656, 0.8462578,  0.8462578,  0.96989137, 0.96815336, 0.96815336,
             0.54361,    0.96815336, 0.96815336, 0.8556079,  0.9142884,  0.9142884,  0.8556079,  0.98927075, 0.99933624, 0.6295072,
             0.63463855, 0.99933624, 0.98111194, 0.82608557, 0.6107712,  0.98111194, 0.82608557, 0.7088349,  0.6449191,  0.7088349,
             0.7088349,  0.8171236,  0.7166788,  0.5830564,  0.8171236,  0.9575846,  0.9575846,  0.10768154, 0.9575846,  0.9575846,
             0.86659664, 0.1641238,  0.94771904, 0.86659664, 0.8357075,  0.7981461,  0.8357075,  0.8357075,  0.09764841, 0.9327884,
             0.9327884,  0.32184523, 0.87756634, 0.87756634, 0.6356852,  0.87756634, 0.87756634, 0.6356852});  // Computed with Chainer.

    testing::ExpectEqual(e_out, out);
}

TEST_P(PoolingTest, MaxPoolingCoverAll) {
    if (GetParam() == "cuda") {
        // TODO(hvy): Test CUDA when implemented.
        return;
    }
    using T = float;

    int64_t batch_size = 3;
    int64_t channels = 4;
    Shape in_dims{4, 4};
    StackVector<int64_t, kMaxNdim> kernel_size{3, 2};
    StackVector<int64_t, kMaxNdim> stride{2, 1};
    StackVector<int64_t, kMaxNdim> pad{1, 0};
    Shape out_dims{2, 3};
    bool cover_all = false;

    Shape x_shape{batch_size, channels};
    std::copy(in_dims.begin(), in_dims.end(), std::back_inserter(x_shape));
    Shape out_shape{batch_size, channels};
    std::copy(out_dims.begin(), out_dims.end(), std::back_inserter(out_shape));

    Array x = testing::BuildArray(x_shape)
                      .WithData<T>(
                              {0.951628,   0.8341918,  0.5700014,  0.02573909, 0.10652946, 0.45143777, 0.12487986, 0.6040584,  0.7059066,
                               0.8674204,  0.89753157, 0.3271186,  0.22637007, 0.7894245,  0.9550997,  0.03499391, 0.6357232,  0.4714537,
                               0.26333022, 0.15744655, 0.69606817, 0.5913821,  0.1362648,  0.00700566, 0.6983082,  0.41985217, 0.19198065,
                               0.87712926, 0.01699107, 0.85048497, 0.6478966,  0.81732035, 0.47958362, 0.335237,   0.6713821,  0.1833262,
                               0.8953133,  0.8300278,  0.46769994, 0.76619476, 0.57752323, 0.60258865, 0.27085522, 0.79189676, 0.98663867,
                               0.50531614, 0.16972028, 0.9301859,  0.53940713, 0.42947277, 0.7620938,  0.4948149,  0.2600685,  0.8730976,
                               0.3494606,  0.9889337,  0.5368636,  0.4020234,  0.23665707, 0.41831595, 0.62009174, 0.7569111,  0.7489499,
                               0.60345465, 0.8451688,  0.84799254, 0.99623865, 0.0536505,  0.320729,   0.68655115, 0.9852334,  0.890243,
                               0.76959133, 0.3614867,  0.11742796, 0.7991817,  0.05568137, 0.22353998, 0.26920173, 0.5037702,  0.45541075,
                               0.45879447, 0.48008284, 0.57052517, 0.3782304,  0.637869,   0.45500278, 0.71749103, 0.9862718,  0.21877514,
                               0.10590941, 0.5953773,  0.46771872, 0.73789245, 0.4005024,  0.7518998,  0.9913527,  0.5310464,  0.4475842,
                               0.5483692,  0.965521,   0.8801182,  0.18907578, 0.95214474, 0.02703529, 0.51783687, 0.17790386, 0.40175965,
                               0.5297797,  0.7417257,  0.22830275, 0.5155725,  0.933218,   0.31846902, 0.9928533,  0.8593246,  0.8691987,
                               0.83518404, 0.69086736, 0.3735951,  0.65166473, 0.58173877, 0.8519384,  0.54010224, 0.03113064, 0.4510318,
                               0.674089,   0.76923084, 0.42310983, 0.31675196, 0.8791844,  0.01504437, 0.98128337, 0.8053975,  0.14322701,
                               0.9443598,  0.96856105, 0.46812293, 0.6314993,  0.6479647,  0.44749212, 0.9877724,  0.7250273,  0.49135047,
                               0.56493795, 0.6489228,  0.04269254, 0.20499802, 0.16736922, 0.7334596,  0.40343535, 0.06048108, 0.7591618,
                               0.63597775, 0.11817221, 0.2982908,  0.00329836, 0.27108955, 0.02329292, 0.69136006, 0.8659653,  0.24925236,
                               0.33170977, 0.02298746, 0.11057855, 0.06332088, 0.04107838, 0.86021507, 0.72832036, 0.44712546, 0.15952812,
                               0.44132948, 0.8370784,  0.46001586, 0.14595562, 0.18176174, 0.68951994, 0.37592548, 0.0262325,  0.40434295,
                               0.05052375, 0.05624698, 0.10016874, 0.9320143,  0.09351984, 0.53812116, 0.20279366, 0.22279656, 0.33266315,
                               0.8101899,  0.6632538,  0.64406633})
                      .WithPadding(1);  // Computed with Chainer.

    Array out = MaxPooling(x, kernel_size, stride, pad, cover_all);

    Array e_out = testing::BuildArray(out_shape).WithData<T>(
            {0.951628,   0.8341918,  0.6040584,  0.8674204, 0.9550997,  0.9550997,  0.69606817, 0.5913821,  0.26333022, 0.85048497,
             0.85048497, 0.87712926, 0.8953133,  0.8300278, 0.76619476, 0.98663867, 0.8300278,  0.9301859,  0.8730976,  0.8730976,
             0.9889337,  0.8730976,  0.8730976,  0.9889337, 0.84799254, 0.99623865, 0.99623865, 0.76959133, 0.9852334,  0.9852334,
             0.637869,   0.637869,   0.71749103, 0.9862718, 0.73789245, 0.7518998,  0.9913527,  0.8801182,  0.95214474, 0.965521,
             0.8801182,  0.95214474, 0.933218,   0.9928533, 0.9928533,  0.8691987,  0.8519384,  0.8519384,  0.98128337, 0.8791844,
             0.9443598,  0.9877724,  0.9877724,  0.9443598, 0.7334596,  0.7334596,  0.40343535, 0.7591618,  0.7334596,  0.69136006,
             0.8659653,  0.33170977, 0.86021507, 0.8370784, 0.46001586, 0.86021507, 0.68951994, 0.37592548, 0.9320143,  0.8101899,
             0.8101899,  0.9320143});  // Computed with Chainer.

    testing::ExpectEqual(e_out, out);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        PoolingTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
