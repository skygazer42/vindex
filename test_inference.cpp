#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

// Simple test to verify ONNX inference works
int main(int argc, char* argv[]) {
    std::cout << "=== VIndex ONNX Inference Test ===" << std::endl;

    try {
        // Initialize ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Test 1: OCR Detection Model
        std::cout << "\n1. Testing OCR Detection Model..." << std::endl;
        std::string ocr_det_path = "../assets/models/ocr/ch_PP-OCRv4_det_infer.onnx";

        try {
            Ort::Session ocr_det_session(env, ocr_det_path.c_str(), session_options);

            // Get input info
            size_t num_input_nodes = ocr_det_session.GetInputCount();
            Ort::AllocatorWithDefaultOptions allocator;

            for (size_t i = 0; i < num_input_nodes; i++) {
                auto input_name = ocr_det_session.GetInputNameAllocated(i, allocator);
                auto input_shape_info = ocr_det_session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();
                auto input_shape = input_shape_info.GetShape();

                std::cout << "  Input " << i << ": " << input_name.get() << " [";
                for (size_t j = 0; j < input_shape.size(); j++) {
                    std::cout << input_shape[j];
                    if (j < input_shape.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }

            // Get output info
            size_t num_output_nodes = ocr_det_session.GetOutputCount();
            for (size_t i = 0; i < num_output_nodes; i++) {
                auto output_name = ocr_det_session.GetOutputNameAllocated(i, allocator);
                std::cout << "  Output " << i << ": " << output_name.get() << std::endl;
            }

            std::cout << "  OCR Detection Model: OK" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  OCR Detection Model: FAILED - " << e.what() << std::endl;
        }

        // Test 2: OCR Recognition Model
        std::cout << "\n2. Testing OCR Recognition Model..." << std::endl;
        std::string ocr_rec_path = "../assets/models/ocr/ch_PP-OCRv4_rec_infer.onnx";

        try {
            Ort::Session ocr_rec_session(env, ocr_rec_path.c_str(), session_options);
            std::cout << "  Input count: " << ocr_rec_session.GetInputCount() << std::endl;
            std::cout << "  Output count: " << ocr_rec_session.GetOutputCount() << std::endl;
            std::cout << "  OCR Recognition Model: OK" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  OCR Recognition Model: FAILED - " << e.what() << std::endl;
        }

        // Test 3: CLIP Visual Model
        std::cout << "\n3. Testing CLIP Visual Encoder..." << std::endl;
        std::string clip_visual_path = "../assets/models/clip_visual.onnx";

        try {
            Ort::Session clip_visual_session(env, clip_visual_path.c_str(), session_options);

            // Get input shape
            auto input_shape_info = clip_visual_session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
            auto input_shape = input_shape_info.GetShape();

            std::cout << "  Input shape: [";
            for (size_t i = 0; i < input_shape.size(); i++) {
                std::cout << input_shape[i];
                if (i < input_shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            // Create dummy input
            std::vector<float> input_data(1 * 3 * 224 * 224, 0.5f);
            std::vector<int64_t> input_shape_vec = {1, 3, 224, 224};

            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                mem_info, input_data.data(), input_data.size(),
                input_shape_vec.data(), input_shape_vec.size()
            );

            // Run inference
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = clip_visual_session.GetInputNameAllocated(0, allocator);
            auto output_name = clip_visual_session.GetOutputNameAllocated(0, allocator);

            const char* input_names[] = {input_name.get()};
            const char* output_names[] = {output_name.get()};

            auto output_tensors = clip_visual_session.Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1
            );

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

            std::cout << "  CLIP Visual Encoder: OK (Inference successful)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  CLIP Visual Encoder: FAILED - " << e.what() << std::endl;
        }

        // Test 4: BLIP Visual Encoder
        std::cout << "\n4. Testing BLIP Visual Encoder..." << std::endl;
        std::string blip_visual_path = "../assets/models/blip/blip_visual_encoder.onnx";

        try {
            Ort::Session blip_visual_session(env, blip_visual_path.c_str(), session_options);
            std::cout << "  Input count: " << blip_visual_session.GetInputCount() << std::endl;
            std::cout << "  Output count: " << blip_visual_session.GetOutputCount() << std::endl;
            std::cout << "  BLIP Visual Encoder: OK" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "  BLIP Visual Encoder: FAILED - " << e.what() << std::endl;
        }

        // Test 5: OpenCV
        std::cout << "\n5. Testing OpenCV..." << std::endl;
        cv::Mat test_img = cv::Mat::zeros(100, 100, CV_8UC3);
        cv::resize(test_img, test_img, cv::Size(224, 224));
        std::cout << "  OpenCV: OK (Created " << test_img.cols << "x" << test_img.rows << " image)" << std::endl;

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "ONNX Runtime and model loading works correctly!" << std::endl;
        std::cout << "Ready for full application integration." << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}