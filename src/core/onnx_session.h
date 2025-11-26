#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <memory>

namespace vindex {
namespace core {

/**
 * @brief ONNX Runtime 会话轻量包装
 *
 * 统一会话创建配置，方便未来共享 Ort::Env。
 */
class OnnxSession {
public:
    OnnxSession();
    ~OnnxSession() = default;

    /**
     * @brief 加载模型
     * @param env 共享 ORT 环境
     * @param modelPath 模型路径
     */
    void load(Ort::Env& env, const std::string& modelPath);

    /**
     * @brief 获取底层会话
     */
    Ort::Session* get() { return session_.get(); }
    const Ort::Session* get() const { return session_.get(); }

    /**
     * @brief 是否已加载
     */
    bool loaded() const { return session_ != nullptr; }

private:
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;
};

} // namespace core
} // namespace vindex
