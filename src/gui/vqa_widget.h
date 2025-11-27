#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QGroupBox>
#include <QString>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief BLIP VQA 视觉问答界面
 *
 * 使用 Taiyi-BLIP-750M-Chinese 模型进行中文视觉问答
 */
class VQAWidget : public QWidget {
    Q_OBJECT
public:
    explicit VQAWidget(core::ModelManager* modelManager,
                       QWidget* parent = nullptr);
    ~VQAWidget() = default;

private slots:
    void onSelectImage();
    void onAsk();
    void retranslateUI();

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QGroupBox* inputGroup_;
    QGroupBox* questionGroup_;
    QGroupBox* outputGroup_;
    QLabel* imageLabel_;
    QLineEdit* questionEdit_;
    QLabel* questionLabel_;
    QLabel* answerLabel_;
    QPushButton* selectBtn_;
    QPushButton* askBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
