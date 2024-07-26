#include <torch/torch.h>
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


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

torch::Tensor read_image(const std::string& image_path) 
{
    int width, height, channels;
    unsigned char* img = stbi_load(image_path.c_str(), &width, &height, &channels, 1);

    if (img == nullptr) 
    {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    torch::Tensor img_tensor = torch::from_blob(img, { 1, height, width }, torch::kUInt8);
    img_tensor = img_tensor.to(torch::kFloat32).div(255.0);
    stbi_image_free(img);

    return img_tensor;
}

class CustomDataset : public torch::data::Dataset<CustomDataset> 
{
public:
    CustomDataset(const std::string& root_dir, const std::vector<std::string>& classes)//接受根目录和类别列表
        : root_dir_(root_dir), classes_(classes) 
    {
        for (size_t label = 0; label < classes.size(); ++label) 
        {
            for (const auto& entry : std::filesystem::directory_iterator(root_dir + "/" + classes[label])) 
            {
                image_paths_.emplace_back(entry.path().string(), label);//将图像路径和标签存入image_paths_
            }
        }
        std::cout << "Dataset loaded with " << image_paths_.size() << " images." << std::endl;
    }

    torch::data::Example<> get(size_t index) override 
    {
        auto image_path = image_paths_[index].first;//获取图像路径和标签
        auto label = image_paths_[index].second;//获取标签
        return { read_image(image_path), torch::tensor(label, torch::kInt64) };//返回图像和标签
    }

    torch::optional<size_t> size() const override 
    {
        return image_paths_.size();//返回数据集大小
    }

private:
    std::string root_dir_;//根目录
    std::vector<std::string> classes_;//类别列表
    std::vector<std::pair<std::string, int>> image_paths_;//图像路径和标签列表
};

int main() 
{
    try 
    {
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);//选择设备
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;//输出设备类型

        const std::string train_root = "D:/OPENCV/data/train";//训练集根目录
        const std::string test_root = "D:/OPENCV/data/test";//测试集根目录
        const std::vector<std::string> train_classes = { "surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry" };
        const std::vector<std::string> test_classes = { "surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry" };
        //"angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"
        //"surprised", "sad", "neutral", "happy", "fearful", "disgusted", "angry"
        auto train_dataset = CustomDataset(train_root, train_classes).map(torch::data::transforms::Stack<>());
        //stack新增批次维度比如64*1*48*48
        //std::cout << "train_dataset size is :" << train_dataset.size().value() << std::endl;
        auto test_dataset = CustomDataset(test_root, test_classes).map(torch::data::transforms::Stack<>());

        auto train_loader = torch::data::make_data_loader(train_dataset, 100);
        auto test_loader = torch::data::make_data_loader(test_dataset, 100);

        auto net = std::make_shared<Net>();
        net->to(device);

        torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.004));

        for (size_t epoch = 1; epoch <= 50; ++epoch) 
        {
            net->train();
            size_t batch_index = 0;
            double epoch_loss = 0.0;
            size_t total_batches = 0;

            for (auto& batch : *train_loader) 
            {
                optimizer.zero_grad();
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                //std::cout << "Training on device: " << (data.device().is_cuda() ? "CUDA" : "CPU") << std::endl;
                auto output = net->forward(data).to(device);
                auto loss = torch::nll_loss(output, target).to(device);
                loss.backward();
                optimizer.step();

                epoch_loss += loss.template item<double>();
                total_batches++;

                if (batch_index % 10 == 0) 
                {
                    std::cout << "Train Epoch: " << epoch << " [" << batch_index * batch.data.size(0)
                        << "/" << train_dataset.size().value() << "]\tBatch Loss: " << loss.template item<float>() << std::endl;
                }
                ++batch_index;
            }

            double avg_epoch_loss = epoch_loss / total_batches;
            std::cout << "Epoch: " << epoch << " Average Loss: " << avg_epoch_loss << std::endl;

            // Evaluate on test set
            net->eval();
            torch::NoGradGuard no_grad;
            double test_loss = 0.0;
            size_t correct = 0;
            size_t total_test_samples = 0;

            for (const auto& batch : *test_loader) 
            {
                auto data = batch.data.to(device);
                auto target = batch.target.to(device);
                auto output = net->forward(data).to(device);
                test_loss += torch::nll_loss(output, target, {}, torch::Reduction::Sum).template item<double>();
                auto pred = output.argmax(1);
                correct += pred.eq(target).sum().template item<int>();
                total_test_samples += data.size(0);
                //data.size(0) 可以用来确定当前批次包含的样本数量
            }

            test_loss /= total_test_samples;
            double accuracy = static_cast<double>(correct) / total_test_samples;

            std::cout << "Test set: Average loss: " << test_loss
                << ", Accuracy: " << correct << "/" << total_test_samples
                << " (" << std::fixed << std::setprecision(2) << (100.0 * accuracy) << "%)\n";
            //输出小数点后两位
        }

        torch::save(net, "D:/E3.pt");
        std::cout << "Model saved to D:/emotion_classification_model.pt" << std::endl;
    }
    catch (const std::exception& e) 
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
