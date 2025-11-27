#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QGroupBox>
#include <QString>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief BLIP 图像描述界面
 *
 * 使用 Taiyi-BLIP-750M-Chinese 模型生成中文图像描述
 */
class CaptionWidget : public QWidget {
    Q_OBJECT
public:
    explicit CaptionWidget(core::ModelManager* modelManager,
                           QWidget* parent = nullptr);
    ~CaptionWidget() = default;

private slots:
    void onSelectImage();
    void onGenerate();
    void retranslateUI();

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QGroupBox* inputGroup_;
    QGroupBox* outputGroup_;
    QLabel* imageLabel_;
    QLabel* captionLabel_;
    QPushButton* selectBtn_;
    QPushButton* generateBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
