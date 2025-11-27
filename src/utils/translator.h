#pragma once

#include <QObject>
#include <QTranslator>
#include <QMap>
#include <QString>
#include <memory>

namespace vindex {
namespace utils {

/**
 * @brief 翻译管理器 - 支持中英文切换
 */
class Translator : public QObject {
    Q_OBJECT

public:
    enum Language {
        English,
        Chinese
    };

    static Translator& instance();

    void setLanguage(Language lang);
    Language currentLanguage() const { return currentLang_; }
    QString languageName(Language lang) const;

    // 获取翻译文本
    static QString translate(const char* key);

signals:
    void languageChanged();

private:
    Translator();
    ~Translator() = default;
    Translator(const Translator&) = delete;
    Translator& operator=(const Translator&) = delete;

    void loadTranslations();

    Language currentLang_;
    QMap<QString, QString> zhTranslations_;
};

// 便捷宏
#define TR(key) vindex::utils::Translator::translate(key)

} // namespace utils
} // namespace vindex
