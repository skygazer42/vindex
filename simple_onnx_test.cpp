#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <filesystem>

// Windows-specific includes for DLL loading
#ifdef _WIN32
#include <windows.h>
#endif

// ONNX Runtime C++ API
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

void print_model_info(const std::string& model_name, const std::string& model_path, Ort::Session& session) {
    std::cout << "\n[" << model_name << "]" << std::endl;
    std::cout << "  Path: " << model_path << std::endl;

    // Get input info
    size_t num_inputs = session.GetInputCount();
    std::cout << "  Inputs: " << num_inputs << std::endl;

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < num_inputs; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        auto input_shape_info = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
        auto input_shape = input_shape_info.GetShape();

        std::cout << "    [" << i << "] " << input_name.get() << ": [";
        for (size_t j = 0; j < input_shape.size(); j++) {
            std::cout << input_shape[j];
            if (j < input_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }

    // Get output info
    size_t num_outputs = session.GetOutputCount();
    std::cout << "  Outputs: " << num_outputs << std::endl;
    for (size_t i = 0; i < num_outputs; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        std::cout << "    [" << i << "] " << output_name.get() << std::endl;
    }
}

bool test_inference(const std::string& model_name, const std::string& model_path,
                   Ort::Session& session, const std::vector<int64_t>& input_shape) {
    try {
        std::cout << "  Running inference test..." << std::endl;

        // Calculate input size
        size_t input_size = 1;
        for (auto dim : input_shape) {
            input_size *= dim;
        }

        // Create dummy input data
        std::vector<float> input_data(input_size, 0.5f);

        // Create input tensor
        Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            mem_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session.GetInputNameAllocated(0, allocator);
        auto output_name = session.GetOutputNameAllocated(0, allocator);

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};

        // Run inference
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names, &input_tensor, 1,
            output_names, 1);

        // Check output
        auto& output = output_tensors[0];
        auto output_shape_info = output.GetTensorTypeAndShapeInfo();
        auto output_shape = output_shape_info.GetShape();

        std::cout << "  Output shape: [";
        for (size_t i = 0; i < output_shape.size(); i++) {
            std::cout << output_shape[i];
            if (i < output_shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "  ✓ Inference successful!" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cout << "  ✗ Inference failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== VIndex ONNX C++ Inference Test ===" << std::endl;
    std::cout << "Testing ONNX Runtime integration without Qt/OpenCV dependencies\n" << std::endl;

    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Model paths and test configurations
        struct TestCase {
            std::string name;
            std::string path;
            std::vector<int64_t> input_shape;
            bool test_inference;
        };

        std::vector<TestCase> test_cases = {
            {"OCR Detection", "assets/models/ocr/ch_PP-OCRv4_det_infer.onnx", {1, 3, 640, 640}, true},
            {"OCR Recognition", "assets/models/ocr/ch_PP-OCRv4_rec_infer.onnx", {1, 3, 48, 320}, true},
            {"CLIP Visual", "assets/models/clip_visual.onnx", {1, 3, 224, 224}, true},
            {"BLIP Visual", "assets/models/blip/blip_visual_encoder.onnx", {1, 3, 384, 384}, true},
            {"BLIP Text Decoder", "assets/models/blip/blip_text_decoder.onnx", {}, false},  // Complex inputs
            {"VQA Visual", "assets/models/blip_vqa/blip_vqa_visual_encoder.onnx", {1, 3, 384, 384}, true}
        };

        int success_count = 0;
        int total_count = test_cases.size();
        std::vector<std::pair<std::string, bool>> results;

        for (const auto& test : test_cases) {
            std::cout << "\n--- Testing " << test.name << " ---" << std::endl;

            // Check if file exists
            if (!fs::exists(test.path)) {
                std::cout << "  ✗ Model file not found: " << test.path << std::endl;
                results.push_back({test.name, false});
                continue;
            }

            // Get file size
            auto file_size = fs::file_size(test.path);
            std::cout << "  File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;

            try {
                // Load model
                #ifdef _WIN32
                std::wstring wpath(test.path.begin(), test.path.end());
                Ort::Session session(env, wpath.c_str(), session_options);
                #else
                Ort::Session session(env, test.path.c_str(), session_options);
                #endif

                // Print model info
                print_model_info(test.name, test.path, session);

                // Test inference if requested
                bool test_passed = true;
                if (test.test_inference && !test.input_shape.empty()) {
                    test_passed = test_inference(test.name, test.path, session, test.input_shape);
                } else if (!test.test_inference) {
                    std::cout << "  ✓ Model loaded successfully (complex inputs, skipping inference test)" << std::endl;
                }

                if (test_passed) {
                    success_count++;
                    results.push_back({test.name, true});
                } else {
                    results.push_back({test.name, false});
                }

            } catch (const Ort::Exception& e) {
                std::cout << "  ✗ ONNX Runtime error: " << e.what() << std::endl;
                results.push_back({test.name, false});
            } catch (const std::exception& e) {
                std::cout << "  ✗ Error: " << e.what() << std::endl;
                results.push_back({test.name, false});
            }
        }

        // Print summary
        std::cout << "\n\n=== Summary ===" << std::endl;
        for (const auto& [name, success] : results) {
            std::cout << "  " << (success ? "✓" : "✗") << " " << name << std::endl;
        }

        std::cout << "\nPassed: " << success_count << "/" << total_count << std::endl;

        if (success_count == total_count) {
            std::cout << "\n✓ All models working correctly in C++!" << std::endl;
            std::cout << "Ready for full application integration." << std::endl;
        } else if (success_count > 0) {
            std::cout << "\n" << success_count << " models working in C++." << std::endl;
            std::cout << "Some issues need to be resolved." << std::endl;
        } else {
            std::cout << "\n✗ No models working. Please check ONNX Runtime installation." << std::endl;
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime initialization error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}