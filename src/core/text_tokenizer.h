#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace vindex {
namespace core {

/**
 * @brief CLIP BPE文本分词器
 *
 * 实现CLIP使用的Byte-Pair Encoding (BPE)分词算法
 * 参考：https://github.com/openai/CLIP/blob/main/clip/simple_tokenizer.py
 */
class TextTokenizer {
public:
    /**
     * @brief 构造函数
     * @param bpeVocabPath BPE词表文件路径 (bpe_simple_vocab_16e6.txt)
     * @param contextLength 上下文长度（CLIP默认77）
     */
    explicit TextTokenizer(const std::string& bpeVocabPath, int contextLength = 77);
    ~TextTokenizer() = default;

    /**
     * @brief 将文本转换为token IDs
     * @param text 输入文本
     * @return token IDs向量 (长度固定为contextLength)
     */
    std::vector<int64_t> encode(const std::string& text);

    /**
     * @brief 批量编码文本
     * @param texts 文本列表
     * @return token IDs矩阵 (batch_size, contextLength)
     */
    std::vector<int64_t> encodeBatch(const std::vector<std::string>& texts);

    /**
     * @brief 将token IDs解码为文本
     * @param tokens token IDs
     * @return 解码后的文本
     */
    std::string decode(const std::vector<int64_t>& tokens);

    int getContextLength() const { return contextLength_; }
    int getVocabSize() const { return vocabSize_; }

private:
    struct PairHash {
        size_t operator()(const std::pair<std::string, std::string>& p) const {
            return std::hash<std::string>()(p.first) ^ std::hash<std::string>()(p.second);
        }
    };

    /**
     * @brief 加载BPE词表
     */
    void loadVocabulary(const std::string& vocabPath);

    /**
     * @brief 基础文本清理
     */
    std::string cleanText(const std::string& text);

    /**
     * @brief 将文本分割为基础tokens
     */
    std::vector<std::string> basicTokenize(const std::string& text);

    /**
     * @brief BPE编码单个词
     */
    std::vector<std::string> bpeEncode(const std::string& token);

    /**
     * @brief 获取字符的字节表示
     */
    std::vector<int> bytesFromUTF8(const std::string& text);

private:
    int contextLength_;                                    // 上下文长度
    int vocabSize_;                                        // 词表大小
    int sotToken_;                                         // Start of text token ID
    int eotToken_;                                         // End of text token ID

    std::unordered_map<std::string, int> encoder_;         // BPE token -> ID
    std::unordered_map<int, std::string> decoder_;         // ID -> BPE token
    std::unordered_map<std::pair<std::string, std::string>, int, PairHash> bpeMerges_; // BPE合并规则
};

} // namespace core
} // namespace vindex
