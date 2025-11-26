#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include <opencv2/opencv.hpp>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 图文匹配界面
 *
 * 提供加载图像 + 输入文本，计算CLIP相似度。
 * 若文本模型未加载，会提示用户。
 */
class MatchWidget : public QWidget {
    Q_OBJECT
public:
    explicit MatchWidget(core::ModelManager* modelManager,
                         QWidget* parent = nullptr);
    ~MatchWidget() = default;

private slots:
    void onSelectImage();
    void onCompute();

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QLabel* imageLabel_;
    QLineEdit* textEdit_;
    QLabel* scoreLabel_;
    QPushButton* selectBtn_;
    QPushButton* computeBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
