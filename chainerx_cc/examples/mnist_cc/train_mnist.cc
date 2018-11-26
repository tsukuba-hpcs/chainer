#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/backprop_mode.h"
#include "chainerx/backward.h"
#include "chainerx/device_id.h"
#include "chainerx/dtype.h"
#include "chainerx/routines/creation.h"
#include "chainerx/routines/linalg.h"
#include "chainerx/routines/manipulation.h"
#include "chainerx/routines/math.h"
#include "chainerx/shape.h"
#include "chainerx/slice.h"

int32_t ReverseInt(int32_t i) {
    uint8_t c1 = i & 255;
    uint8_t c2 = (i >> 8) & 255;
    uint8_t c3 = (i >> 16) & 255;
    uint8_t c4 = (i >> 24) & 255;

    return (static_cast<int32_t>(c1) << 24) + (static_cast<int32_t>(c2) << 16) + (static_cast<int32_t>(c3) << 8) + c4;
}

template <typename T>
chainerx::Array GetRandomArray(std::mt19937& gen, std::normal_distribution<T>& dist, const chainerx::Shape& shape) {
    int64_t n = shape.GetTotalSize();
    std::shared_ptr<T> data{new T[n], std::default_delete<T[]>{}};
    std::generate_n(data.get(), n, [&dist, &gen]() { return dist(gen); });
    return chainerx::FromContiguousHostData(
            shape, chainerx::TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data), chainerx::GetDefaultDevice());
}

template <typename T>
chainerx::Array ReadMnistData(std::ifstream& ifs, const chainerx::Shape& shape) {
    int64_t n = shape.GetTotalSize();
    std::shared_ptr<T> data{new T[n], std::default_delete<T[]>{}};
    std::generate_n(data.get(), n, [&ifs]() {
        T d = 0;
        ifs.read(reinterpret_cast<char*>(&d), sizeof(d));
        return d;
    });
    return chainerx::FromContiguousHostData(
            shape, chainerx::TypeToDtype<T>, static_cast<std::shared_ptr<void>>(data), chainerx::GetDefaultDevice());
}

chainerx::Array ReadMnistImages(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not read MNIST images from " + filename);
    }

    ifs.ignore(sizeof(int32_t));  // Ignore the magic number.
    int32_t n{0};
    int32_t height{0};
    int32_t width{0};
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    ifs.read(reinterpret_cast<char*>(&height), sizeof(height));
    ifs.read(reinterpret_cast<char*>(&width), sizeof(width));
    n = ReverseInt(n);
    height = ReverseInt(height);
    width = ReverseInt(width);

    return ReadMnistData<uint8_t>(ifs, {n, height * width});
}

chainerx::Array ReadMnistLabels(const std::string& filename) {
    std::ifstream ifs(filename.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not read MNIST labels from " + filename);
    }

    ifs.ignore(sizeof(int32_t));  // Ignore the magic number.
    int32_t n{0};
    ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
    n = ReverseInt(n);

    return ReadMnistData<uint8_t>(ifs, {n});
}

class Model {
public:
    Model(int64_t n_in, int64_t n_hidden, int64_t n_out, int64_t n_layers)
        : n_in_{n_in}, n_hidden_{n_hidden}, n_out_{n_out}, n_layers_{n_layers} {};

    chainerx::Array operator()(const chainerx::Array& x) {
        chainerx::Array h = x;
        for (int64_t i = 0; i < n_layers_; ++i) {
            h = chainerx::Dot(h, params_[i * 2]) + params_[i * 2 + 1];
            if (i != n_layers_ - 1) {
                h = chainerx::Maximum(0, h);
            }
        }
        return h;
    }

    const std::vector<chainerx::Array>& params() { return params_; }

    template <typename T>
    void Initialize(std::mt19937& gen, std::normal_distribution<T>& dist) {
        params_.clear();

        for (int64_t i = 0; i < n_layers_; ++i) {
            int64_t n_in = i == 0 ? n_in_ : n_hidden_;
            int64_t n_out = i == n_layers_ - 1 ? n_out_ : n_hidden_;
            params_.emplace_back(GetRandomArray(gen, dist, {n_in, n_out}));
            params_.emplace_back(chainerx::Zeros({n_out}, chainerx::TypeToDtype<T>));
        }

        for (const chainerx::Array& param : params_) {
            param.RequireGrad();
        }
    }

private:
    int64_t n_in_;
    int64_t n_hidden_;
    int64_t n_out_;
    int64_t n_layers_;
    std::vector<chainerx::Array> params_;
};

chainerx::Array SoftmaxCrossEntropy(const chainerx::Array& y, const chainerx::Array& t, bool normalize = true) {
    chainerx::Array score = chainerx::LogSoftmax(y, 1);
    chainerx::Array mask =
            (t.At({chainerx::Slice{}, chainerx::NewAxis{}}) == chainerx::Arange(score.shape()[1], t.dtype())).AsType(score.dtype());
    if (normalize) {
        return -(score * mask).Sum() / y.shape()[0];
    }
    return -(score * mask).Sum();
}

void RunWithDefaultDevice(int64_t epochs, int64_t batch_size, int64_t n_hidden, int64_t n_layers, float lr, const std::string& mnist_root) {
    // Read the MNIST dataset.
    chainerx::Array train_x = ReadMnistImages(mnist_root + "train-images-idx3-ubyte");
    chainerx::Array train_t = ReadMnistLabels(mnist_root + "train-labels-idx1-ubyte");
    chainerx::Array test_x = ReadMnistImages(mnist_root + "t10k-images-idx3-ubyte");
    chainerx::Array test_t = ReadMnistLabels(mnist_root + "t10k-labels-idx1-ubyte");

    train_x = train_x.AsType(chainerx::Dtype::kFloat32) / 255.f;
    train_t = train_t.AsType(chainerx::Dtype::kInt32);
    test_x = test_x.AsType(chainerx::Dtype::kFloat32) / 255.f;
    test_t = test_t.AsType(chainerx::Dtype::kInt32);

    int64_t n_train = train_x.shape().front();
    int64_t n_test = test_x.shape().front();
    chainerx::Array train_indices = chainerx::Arange(n_train, chainerx::Dtype::kInt64);

    // Initialize the model with random parameters.
    Model model{train_x.shape()[1], n_hidden, 10, n_layers};
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> dist{0.f, 0.05f};
    model.Initialize(gen, dist);

    auto start = std::chrono::high_resolution_clock::now();

    for (int64_t epoch = 0; epoch < epochs; ++epoch) {
        // The underlying data of the indices' Array is contiguous so we can use std::shuffle to randomize the order.
        assert(train_indices.IsContiguous());
        std::shuffle(
                reinterpret_cast<int64_t*>(train_indices.raw_data()), reinterpret_cast<int64_t*>(train_indices.raw_data()) + n_train, gen);
        for (int64_t i = 0; i < n_train; i += batch_size) {
            chainerx::Array indices = train_indices.At({chainerx::Slice{i, i + batch_size}});
            chainerx::Array x = train_x.Take(indices, 0);
            chainerx::Array t = train_t.Take(indices, 0);

            chainerx::Backward(SoftmaxCrossEntropy(model(x), t));

            // Vanilla SGD.
            for (const chainerx::Array& p : model.params()) {
                chainerx::Array view = p.AsGradStopped(chainerx::CopyKind::kView);
                view -= p.GetGrad()->AsGradStopped() * lr;
                p.ClearGrad();
            }
        }

        // Evaluate.
        {
            chainerx::NoBackpropModeScope scope{};

            chainerx::Array loss = chainerx::Zeros({}, chainerx::Dtype::kFloat32);
            chainerx::Array acc = chainerx::Zeros({}, chainerx::Dtype::kFloat32);

            for (int64_t i = 0; i < n_test; i += batch_size) {
                std::vector<chainerx::ArrayIndex> indices{chainerx::Slice{i, i + batch_size}};
                chainerx::Array x = test_x.At(indices);
                chainerx::Array t = test_t.At(indices);
                chainerx::Array y = model(x);
                loss += SoftmaxCrossEntropy(y, t, false);
                acc += (y.ArgMax(1).AsType(t.dtype()) == t).Sum().AsType(acc.dtype());
            }

            std::cout << "epoch: " << epoch << " loss=" << chainerx::AsScalar(loss / n_test)
                      << " accuracy=" << chainerx::AsScalar(acc / n_test)
                      << " elapsed_time=" << std::chrono::duration<double>{std::chrono::high_resolution_clock::now() - start}.count()
                      << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    int64_t epochs{20};
    int64_t batch_size{100};
    int64_t n_hidden{1000};
    int64_t n_layers{3};
    float lr{0.01};
    std::string device_name{"native"};
    std::string mnist_root{"./"};

    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "--epoch") {
            epochs = std::atoi(argv[i + 1]);
        } else if (arg == "--batchsize") {
            batch_size = std::atoi(argv[i + 1]);
        } else if (arg == "--n-hidden") {
            n_hidden = std::atoi(argv[i + 1]);
        } else if (arg == "--n-layers") {
            n_layers = std::atoi(argv[i + 1]);
        } else if (arg == "--lr") {
            lr = std::atof(argv[i + 1]);
        } else if (arg == "--device") {
            device_name = argv[i + 1];
        } else if (arg == "--data") {
            mnist_root = argv[i + 1];
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (mnist_root.back() != '/') {
        mnist_root += "/";
    }

    chainerx::Context ctx{};
    chainerx::SetDefaultContext(&ctx);
    chainerx::Device& device = ctx.GetDevice(device_name);
    chainerx::SetDefaultDevice(&device);

    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Minibatch size: " << batch_size << std::endl;
    std::cout << "Hidden neurons: " << n_hidden << std::endl;
    std::cout << "Layers: " << n_layers << std::endl;
    std::cout << "Learning rate: " << lr << std::endl;
    std::cout << "Device: " << device.name() << std::endl;
    std::cout << "MNIST root: " << mnist_root << std::endl;

    RunWithDefaultDevice(epochs, batch_size, n_hidden, n_layers, lr, mnist_root);
}
