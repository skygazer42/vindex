#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QString>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 图生文界面占位
 *
 * 预留接入 BLIP/GIT 模型的界面结构。
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

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QLabel* imageLabel_;
    QLabel* captionLabel_;
    QPushButton* selectBtn_;
    QPushButton* generateBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
