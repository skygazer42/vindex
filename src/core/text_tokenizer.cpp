#include "text_tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cctype>
#include <locale>

namespace vindex {
namespace core {

// UTF-8 辅助函数
namespace {

// 获取 UTF-8 字符的字节数
inline int utf8CharBytes(unsigned char firstByte) {
    if ((firstByte & 0x80) == 0) return 1;        // 0xxxxxxx
    if ((firstByte & 0xE0) == 0xC0) return 2;     // 110xxxxx
    if ((firstByte & 0xF0) == 0xE0) return 3;     // 1110xxxx
    if ((firstByte & 0xF8) == 0xF0) return 4;     // 11110xxx
    return 1;  // 无效字节
}

// 将 UTF-8 字符串拆分为单个字符（每个元素是一个完整的 UTF-8 字符）
std::vector<std::string> splitUtf8Chars(const std::string& text) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < text.size()) {
        int bytes = utf8CharBytes(static_cast<unsigned char>(text[i]));
        if (i + bytes <= text.size()) {
            chars.push_back(text.substr(i, bytes));
        }
        i += bytes;
    }
    return chars;
}

// 从 UTF-8 字符获取 Unicode 码点
uint32_t utf8ToCodepoint(const std::string& utf8Char) {
    if (utf8Char.empty()) return 0;

    unsigned char c = static_cast<unsigned char>(utf8Char[0]);
    uint32_t cp = 0;

    if ((c & 0x80) == 0) {
        cp = c;
    } else if ((c & 0xE0) == 0xC0 && utf8Char.size() >= 2) {
        cp = (c & 0x1F) << 6;
        cp |= (static_cast<unsigned char>(utf8Char[1]) & 0x3F);
    } else if ((c & 0xF0) == 0xE0 && utf8Char.size() >= 3) {
        cp = (c & 0x0F) << 12;
        cp |= (static_cast<unsigned char>(utf8Char[1]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(utf8Char[2]) & 0x3F);
    } else if ((c & 0xF8) == 0xF0 && utf8Char.size() >= 4) {
        cp = (c & 0x07) << 18;
        cp |= (static_cast<unsigned char>(utf8Char[1]) & 0x3F) << 12;
        cp |= (static_cast<unsigned char>(utf8Char[2]) & 0x3F) << 6;
        cp |= (static_cast<unsigned char>(utf8Char[3]) & 0x3F);
    }

    return cp;
}

// 判断是否为空白字符
bool isWhitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

} // anonymous namespace

TextTokenizer::TextTokenizer(const std::string& vocabPath, int contextLength, bool doLowerCase)
    : contextLength_(contextLength)
    , vocabSize_(0)
    , padToken_(-1)
    , unkToken_(-1)
    , clsToken_(-1)
    , sepToken_(-1)
    , doLowerCase_(doLowerCase) {
    loadVocabulary(vocabPath);
}

void TextTokenizer::loadVocabulary(const std::string& vocabPath) {
    std::ifstream file(vocabPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + vocabPath);
    }

    std::string line;
    int tokenId = 0;

    while (std::getline(file, line)) {
        // 去除行尾空白字符
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n' ||
               line.back() == ' ' || line.back() == '\t')) {
            line.pop_back();
        }

        if (!line.empty()) {
            vocab_[line] = tokenId;
            invVocab_[tokenId] = line;
            tokenId++;
        }
    }

    vocabSize_ = tokenId;

    // 查找特殊 token IDs（如果存在则更新）
    auto it = vocab_.find("[PAD]");
    if (it != vocab_.end()) padToken_ = it->second;

    it = vocab_.find("[UNK]");
    if (it != vocab_.end()) unkToken_ = it->second;

    it = vocab_.find("[CLS]");
    if (it != vocab_.end()) clsToken_ = it->second;

    it = vocab_.find("[SEP]");
    if (it != vocab_.end()) sepToken_ = it->second;

    if (vocab_.empty()) {
        throw std::runtime_error("Vocabulary is empty or invalid");
    }

    // 确保必需的特殊符号存在
    if (clsToken_ < 0 || sepToken_ < 0) {
        throw std::runtime_error("Vocabulary missing required [CLS]/[SEP] tokens");
    }
    if (padToken_ < 0) padToken_ = 0;
    if (unkToken_ < 0) unkToken_ = padToken_;
}

std::vector<int64_t> TextTokenizer::encode(const std::string& text) {
    // 初始化为 padding
    std::vector<int64_t> output(contextLength_, padToken_);

    // [CLS] token
    std::vector<int64_t> ids;
    ids.reserve(contextLength_);
    ids.push_back(clsToken_);

    // 基础分词
    std::vector<std::string> basicTokens = basicTokenize(text);

    // WordPiece 分词并转换为 IDs
    for (const auto& token : basicTokens) {
        std::vector<std::string> subTokens = wordpieceTokenize(token);

        for (const auto& subToken : subTokens) {
            auto it = vocab_.find(subToken);
            if (it != vocab_.end()) {
                ids.push_back(it->second);
            } else {
                ids.push_back(unkToken_);
            }

            if (static_cast<int>(ids.size()) >= contextLength_ - 1) {
                break;
            }
        }

        if (static_cast<int>(ids.size()) >= contextLength_ - 1) {
            break;
        }
    }

    // [SEP] token
    ids.push_back(sepToken_);

    // 截断（如果超出长度）
    if (static_cast<int>(ids.size()) > contextLength_) {
        ids.resize(contextLength_);
    }

    // 拷贝到输出
    std::copy(ids.begin(), ids.end(), output.begin());

    return output;
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
        // 跳过特殊 token 和 padding
        if (tokenId == clsToken_ || tokenId == sepToken_ ||
            tokenId == padToken_) {
            continue;
        }

        auto it = invVocab_.find(static_cast<int>(tokenId));
        if (it != invVocab_.end()) {
            const std::string& token = it->second;
            // 处理 WordPiece 前缀 "##"
            if (token.size() > 2 && token[0] == '#' && token[1] == '#') {
                result += token.substr(2);
            } else {
                if (!result.empty()) {
                    result += " ";
                }
                result += token;
            }
        }
    }

    return result;
}

std::vector<std::string> TextTokenizer::basicTokenize(const std::string& text) {
    std::string cleaned = cleanText(text);
    std::string spaced = tokenizeChineseChars(cleaned);

    std::vector<std::string> tokens;
    std::string current;
    for (const auto& ch : splitUtf8Chars(spaced)) {
        uint32_t cp = utf8ToCodepoint(ch);
        if (isWhitespace(cp)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            continue;
        }
        if (isPunctuation(cp)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            tokens.push_back(ch);
            continue;
        }
        current += ch;
    }
    if (!current.empty()) {
        tokens.push_back(current);
    }

    // 小写与去重音
    if (doLowerCase_) {
        for (auto& tk : tokens) {
            for (char& c : tk) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            tk = stripAccents(tk);
        }
    }

    return tokens;
}

std::vector<std::string> TextTokenizer::wordpieceTokenize(const std::string& token) {
    // 如果整个 token 在词表中，直接返回
    if (vocab_.find(token) != vocab_.end()) {
        return {token};
    }

    std::vector<std::string> pieces;

    // 将 token 拆分为 UTF-8 字符
    std::vector<std::string> chars = splitUtf8Chars(token);

    if (chars.empty()) {
        return {"[UNK]"};
    }

    // 贪心最大匹配 WordPiece 算法
    size_t start = 0;

    while (start < chars.size()) {
        size_t end = chars.size();
        std::string currentSubstr;
        bool found = false;

        while (start < end) {
            // 构建子串（拼接 UTF-8 字符）
            std::string substr;
            for (size_t i = start; i < end; ++i) {
                substr += chars[i];
            }

            // 如果不是第一个子词，添加 "##" 前缀
            if (start > 0) {
                substr = "##" + substr;
            }

            // 查找词表
            if (vocab_.find(substr) != vocab_.end()) {
                currentSubstr = substr;
                found = true;
                break;
            }

            end--;
        }

        if (!found) {
            // 没有找到匹配，整个 token 标记为 [UNK]
            return {"[UNK]"};
        }

        pieces.push_back(currentSubstr);
        start = end;
    }

    return pieces;
}

bool TextTokenizer::isPunctuation(char32_t cp) const {
    if (cp <= 0x7F) {
        unsigned char uc = static_cast<unsigned char>(cp);
        return (uc >= 33 && uc <= 47) ||
               (uc >= 58 && uc <= 64) ||
               (uc >= 91 && uc <= 96) ||
               (uc >= 123 && uc <= 126);
    }
    // Unicode 标点范围
    return (cp >= 0x2000 && cp <= 0x206F) || // General Punctuation
           (cp >= 0x3000 && cp <= 0x303F) || // CJK Symbols and Punctuation
           (cp >= 0xFF00 && cp <= 0xFF65);   // Fullwidth ASCII variants
}

bool TextTokenizer::isChineseChar(uint32_t cp) const {
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Unified Ideographs Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // CJK Unified Ideographs Extension B
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    // CJK Unified Ideographs Extension C
    if (cp >= 0x2A700 && cp <= 0x2B73F) return true;
    // CJK Unified Ideographs Extension D
    if (cp >= 0x2B740 && cp <= 0x2B81F) return true;
    // CJK Unified Ideographs Extension E
    if (cp >= 0x2B820 && cp <= 0x2CEAF) return true;
    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // CJK Compatibility Ideographs Supplement
    if (cp >= 0x2F800 && cp <= 0x2FA1F) return true;
    // CJK Symbols and Punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return true;
    // Fullwidth Forms
    if (cp >= 0xFF00 && cp <= 0xFFEF) return true;

    return false;
}

bool TextTokenizer::isControl(char32_t cp) const {
    if (cp == '\t' || cp == '\n' || cp == '\r') return false;
    if (cp < 32 || (cp >= 0x7F && cp <= 0x9F)) return true;
    return false;
}

bool TextTokenizer::isWhitespace(char32_t cp) const {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == 0x00A0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

std::string TextTokenizer::cleanText(const std::string& text) const {
    std::string out;
    for (const auto& ch : splitUtf8Chars(text)) {
        uint32_t cp = utf8ToCodepoint(ch);
        if (isControl(cp)) continue;
        if (isWhitespace(cp)) {
            out.push_back(' ');
        } else {
            out += ch;
        }
    }
    return out;
}

std::string TextTokenizer::tokenizeChineseChars(const std::string& text) const {
    std::string out;
    for (const auto& ch : splitUtf8Chars(text)) {
        uint32_t cp = utf8ToCodepoint(ch);
        if (isChineseChar(cp)) {
            out += " ";
            out += ch;
            out += " ";
        } else {
            out += ch;
        }
    }
    return out;
}

std::string TextTokenizer::stripAccents(const std::string& text) const {
    // 仅对 ASCII 范围做简单处理：保留原样
    // 可扩展为完整的 unicode 去重音处理
    return text;
}

} // namespace core
} // namespace vindex
