#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QGroupBox>
#include <QString>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief OCR 文字识别界面
 *
 * 使用 PP-OCRv4 模型进行中文文字识别
 */
class OcrWidget : public QWidget {
    Q_OBJECT
public:
    explicit OcrWidget(core::ModelManager* modelManager,
                       QWidget* parent = nullptr);
    ~OcrWidget() = default;

private slots:
    void onSelectImage();
    void onRecognize();
    void onCopyText();
    void retranslateUI();

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QGroupBox* inputGroup_;
    QGroupBox* outputGroup_;
    QLabel* imageLabel_;
    QTextEdit* resultText_;
    QPushButton* selectBtn_;
    QPushButton* recognizeBtn_;
    QPushButton* copyBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
