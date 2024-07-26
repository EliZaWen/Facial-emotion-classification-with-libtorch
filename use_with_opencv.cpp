#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <iomanip>

struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("conv1", torch::nn::Conv2d(1, 32, 3)); // 32 filters of size 3x3
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 3)); // 64 filters of size 3x3
        fc1 = register_module("fc1", torch::nn::Linear(64 * 10 * 10, 500)); // Adjusted input size
        fc2 = register_module("fc2", torch::nn::Linear(500, 7)); // 1024 inputs to 7 outputs
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::max_pool2d(torch::leaky_relu(conv1->forward(x)), 2); // 2x2 max pooling
        x = torch::max_pool2d(torch::leaky_relu(conv2->forward(x)), 2); // 2x2 max pooling
        //std::cout << "Shape before view: " << x.sizes() << std::endl; // Debug print
        x = x.view({ -1, 64 * 10 * 10 }); // Adjusted view
        x = torch::leaky_relu(fc1->forward(x));
        x = torch::dropout(x, 0.2, is_training());
        x = fc2->forward(x);
        return torch::log_softmax(x, 1); // Log softmax
    }

    torch::nn::Conv2d conv1{ nullptr }, conv2{ nullptr };
    torch::nn::Linear fc1{ nullptr }, fc2{ nullptr };
};

torch::Tensor preprocess_frame(const cv::Mat& frame) {
    cv::Mat gray_frame;
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    cv::resize(gray_frame, gray_frame, cv::Size(48, 48));

    auto img_tensor = torch::from_blob(gray_frame.data, { 1, 1, 48, 48 }, torch::kUInt8);
    img_tensor = img_tensor.to(torch::kFloat32).div(255.0);

    return img_tensor;
}

void draw_label(cv::Mat& input_image, std::string label, int left, int top) {
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 0.8;
    int thickness = 2;
    cv::Point textOrg(left, top - 10);
    cv::putText(input_image, label, textOrg, font, font_scale, cv::Scalar(0, 255, 0), thickness);
}

int main() {
    try {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Load the trained model
        auto net = std::make_shared<Net>();
        torch::load(net, "D:/E3.pt");
        net->to(device);
        net->eval();

        // Initialize video capture
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error opening video capture" << std::endl;
            return -1;
        }

        std::vector<std::string> emotion_classes = { "surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry" };
        //"surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry"
        //"Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"
        cv::Mat frame;
        while (true) {
            cap >> frame;
            if (frame.empty()) {
                std::cerr << "Captured empty frame" << std::endl;
                continue;
            }

            // Preprocess the frame
            torch::Tensor img_tensor = preprocess_frame(frame).to(device);

            // Predict emotion
            torch::Tensor output = net->forward(img_tensor);
            auto prediction = output.argmax(1).item<int>();

            // Draw label
            draw_label(frame, emotion_classes[prediction], 30, 30);

            // Display the result
            cv::imshow("Emotion Recognition", frame);

            if (cv::waitKey(1) == 27) {  // Press 'Esc' to exit
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
