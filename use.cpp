#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 32, 5));
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 5));
        fc1 = register_module("fc1", torch::nn::Linear(64 * 9 * 9, 1024));
        fc2 = register_module("fc2", torch::nn::Linear(1024, 7));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::max_pool2d(torch::relu(conv1->forward(x)), 2);
        x = torch::max_pool2d(torch::relu(conv2->forward(x)), 2);
        x = x.view({ -1, 64 * 9 * 9 });
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

torch::Tensor read_image(const std::string& image_path) {
    int width, height, channels;
    unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 1); 

    if (img == nullptr) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    torch::Tensor img_tensor = torch::from_blob(img, { 1, height, width }, torch::kUInt8);
    img_tensor = img_tensor.to(torch::kFloat32).div(255.0); // Normalize to [0, 1]
    stbi_image_free(img);

    return img_tensor;
}

int main() {
    try {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        const std::vector<std::string> classes = { "surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry" };

        auto net = std::make_shared<Net>();
        torch::load(net, "D:/E.pt");
        net->to(device);
        net->eval();

        std::string image_path = "D:/OPENCV/data/test/neutral/im573.png";
        auto image_tensor = read_image(image_path).unsqueeze(0).to(device);

        torch::NoGradGuard no_grad;
        auto output = net->forward(image_tensor);
        auto prediction = output.argmax(1).item<int>();

        std::cout << "Predicted emotion: " << classes[prediction] << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
