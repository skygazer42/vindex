#include "text_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <stdexcept>

namespace vindex {
namespace core {

TextTokenizer::TextTokenizer(const std::string& bpeVocabPath, int contextLength)
    : contextLength_(contextLength)
    , vocabSize_(49408)  // CLIP默认词表大小
    , sotToken_(49406)   // <|startoftext|>
    , eotToken_(49407)   // <|endoftext|>
{
    loadVocabulary(bpeVocabPath);
}

std::vector<int64_t> TextTokenizer::encode(const std::string& text) {
    // 初始化输出向量，全部填充为0（padding）
    std::vector<int64_t> tokens(contextLength_, 0);

    // 添加起始token
    tokens[0] = sotToken_;

    // 清理和分词
    std::string cleanedText = cleanText(text);
    std::vector<std::string> baseTokens = basicTokenize(cleanedText);

    // BPE编码并转换为IDs
    size_t currentPos = 1;  // 从1开始，因为0是SOT token
    for (const auto& token : baseTokens) {
        if (currentPos >= contextLength_ - 1) {  // 保留最后一位给EOT
            break;
        }

        std::vector<std::string> bpeTokens = bpeEncode(token);
        for (const auto& bpeToken : bpeTokens) {
            if (currentPos >= contextLength_ - 1) {
                break;
            }

            auto it = encoder_.find(bpeToken);
            if (it != encoder_.end()) {
                tokens[currentPos++] = it->second;
            }
        }
    }

    // 添加结束token
    tokens[currentPos] = eotToken_;

    return tokens;
}

std::vector<int64_t> TextTokenizer::encodeBatch(const std::vector<std::string>& texts) {
    std::vector<int64_t> allTokens;
    allTokens.reserve(texts.size() * contextLength_);

    for (const auto& text : texts) {
        auto tokens = encode(text);
        allTokens.insert(allTokens.end(), tokens.begin(), tokens.end());
    }

    return allTokens;
}

std::string TextTokenizer::decode(const std::vector<int64_t>& tokens) {
    std::string result;

    for (int64_t tokenId : tokens) {
        // 跳过特殊token和padding
        if (tokenId == sotToken_ || tokenId == eotToken_ || tokenId == 0) {
            continue;
        }

        auto it = decoder_.find(static_cast<int>(tokenId));
        if (it != decoder_.end()) {
            result += it->second;
        }
    }

    return result;
}

void TextTokenizer::loadVocabulary(const std::string& vocabPath) {
    std::ifstream file(vocabPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open BPE vocabulary file: " + vocabPath);
    }

    std::string line;
    int tokenId = 0;

    // 读取BPE词表（格式：每行一个token）
    while (std::getline(file, line) && tokenId < vocabSize_) {
        if (line.empty()) continue;

        // 去除行尾的空白字符
        line.erase(line.find_last_not_of(" \n\r\t") + 1);

        encoder_[line] = tokenId;
        decoder_[tokenId] = line;
        tokenId++;
    }

    if (encoder_.empty()) {
        throw std::runtime_error("Vocabulary is empty or invalid");
    }

    file.close();
}

std::string TextTokenizer::cleanText(const std::string& text) {
    // 转换为小写
    std::string cleaned = text;
    std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);

    // 去除多余的空白字符
    cleaned = std::regex_replace(cleaned, std::regex("\\s+"), " ");

    // 去除首尾空白
    size_t start = cleaned.find_first_not_of(" \t\n\r");
    size_t end = cleaned.find_last_not_of(" \t\n\r");

    if (start == std::string::npos) {
        return "";
    }

    return cleaned.substr(start, end - start + 1);
}

std::vector<std::string> TextTokenizer::basicTokenize(const std::string& text) {
    std::vector<std::string> tokens;

    // 使用正则表达式分割（简化版本）
    // CLIP使用的正则：r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    // 这里使用简化的空格分割
    std::istringstream iss(text);
    std::string token;

    while (iss >> token) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

std::vector<std::string> TextTokenizer::bpeEncode(const std::string& token) {
    // 简化的BPE编码
    // 完整实现需要：
    // 1. 将token分解为字符序列
    // 2. 迭代应用BPE合并规则
    // 3. 返回最终的子词序列

    // 这里返回简化版本：直接将整个token作为一个BPE单元
    std::vector<std::string> result;

    // 如果token在词表中，直接返回
    if (encoder_.find(token) != encoder_.end()) {
        result.push_back(token);
        return result;
    }

    // 否则，按字符分割（简化处理）
    for (char c : token) {
        std::string charStr(1, c);
        result.push_back(charStr);
    }

    return result;
}

std::vector<int> TextTokenizer::bytesFromUTF8(const std::string& text) {
    std::vector<int> bytes;
    bytes.reserve(text.size());

    for (unsigned char c : text) {
        bytes.push_back(static_cast<int>(c));
    }

    return bytes;
}

} // namespace core
} // namespace vindex
