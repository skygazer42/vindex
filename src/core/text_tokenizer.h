#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace vindex {
namespace core {

/**
 * @brief 中文 BERT WordPiece 分词器
 *
 * 专为 CN-CLIP 设计，支持中文文本分词：
 * - 读取 BERT vocab.txt（每行一个 token）
 * - 特殊符号：[CLS] [SEP] [PAD] [UNK]
 * - WordPiece 切分：按最大匹配，子词前缀 "##"
 * - 序列格式：[CLS] + tokens + [SEP] + [PAD]...
 * - 正确处理 UTF-8 中文字符
 */
class TextTokenizer {
public:
    /**
     * @brief 构造函数
     * @param vocabPath 词表文件路径（BERT vocab.txt 格式）
     * @param contextLength 上下文长度（CN-CLIP 默认 52，标准 CLIP 可设 77）
     * @param doLowerCase 是否小写化（英文模型常用 true，中文通常 false）
     */
    explicit TextTokenizer(const std::string& vocabPath, int contextLength = 52, bool doLowerCase = false);
    ~TextTokenizer() = default;

    /**
     * @brief 将文本转换为 token IDs
     * @param text 输入文本
     * @return token IDs 向量 (长度固定为 contextLength)
     */
    std::vector<int64_t> encode(const std::string& text);

    /**
     * @brief 批量编码文本
     * @param texts 文本列表
     * @return token IDs 矩阵 (展平为一维，大小 = batch_size * contextLength)
     */
    std::vector<int64_t> encodeBatch(const std::vector<std::string>& texts);

    /**
     * @brief 将 token IDs 解码为文本
     * @param tokens token IDs
     * @return 解码后的文本
     */
    std::string decode(const std::vector<int64_t>& tokens);

    int getContextLength() const { return contextLength_; }
    int getVocabSize() const { return vocabSize_; }

private:
    /**
     * @brief 加载 BERT 词表
     */
    void loadVocabulary(const std::string& vocabPath);

    std::vector<std::string> basicTokenize(const std::string& text);
    std::vector<std::string> wordpieceTokenize(const std::string& token);
    bool isPunctuation(char32_t cp) const;
    bool isChineseChar(char32_t codepoint) const;
    bool isControl(char32_t cp) const;
    bool isWhitespace(char32_t cp) const;
    std::string cleanText(const std::string& text) const;
    std::string stripAccents(const std::string& text) const;
    std::string tokenizeChineseChars(const std::string& text) const;

private:
    int contextLength_;                                    // 上下文长度 (CN-CLIP: 52)
    int vocabSize_;                                        // 词表大小
    bool doLowerCase_;

    // 特殊 token IDs
    int padToken_;      // [PAD]
    int unkToken_;      // [UNK]
    int clsToken_;      // [CLS]
    int sepToken_;      // [SEP]

    std::unordered_map<std::string, int> vocab_;           // token -> ID
    std::unordered_map<int, std::string> invVocab_;        // ID -> token
};

} // namespace core
} // namespace vindex
