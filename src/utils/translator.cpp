#include "translator.h"
#include <QApplication>

namespace vindex {
namespace utils {

Translator& Translator::instance() {
    static Translator instance;
    return instance;
}

Translator::Translator()
    : currentLang_(English) {
    loadTranslations();
}

void Translator::loadTranslations() {
    // Minimal, UTF-8 safe translations; unknown keys fall back to original text.
    zhTranslations_.clear();
    zhTranslations_["English"] = "English";
    zhTranslations_["Chinese"] = "中文";
    zhTranslations_["VIndex - Visual Search Engine"] = "VIndex - 视觉搜索引擎";
    zhTranslations_["Search"] = "搜索";
    zhTranslations_["Cancel"] = "取消";
    zhTranslations_["OK"] = "确定";
    zhTranslations_["Error"] = "错误";
    zhTranslations_["Loading models..."] = "正在加载模型...";
    zhTranslations_["Model directory not found. Please ensure models are in ./assets/models/"] = "未找到模型目录，请确保模型位于 ./assets/models/";
}

void Translator::setLanguage(Language lang) {
    if (currentLang_ != lang) {
        currentLang_ = lang;
        emit languageChanged();
    }
}

QString Translator::languageName(Language lang) const {
    switch (lang) {
        case English: return "English";
        case Chinese: return "中文";
        default: return "Unknown";
    }
}

QString Translator::translate(const char* key) {
    Translator& t = instance();
    if (t.currentLang_ == English) {
        return QString::fromUtf8(key);
    }

    QString keyStr = QString::fromUtf8(key);
    auto it = t.zhTranslations_.find(keyStr);
    if (it != t.zhTranslations_.end()) {
        return it.value();
    }
    return keyStr; // 未翻译则返回原文
}

} // namespace utils
} // namespace vindex
