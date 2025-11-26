#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QString>
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 图文问答界面占位
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

private:
    void setupUI();
    void showError(const QString& message);

private:
    core::ModelManager* modelManager_;
    QLabel* imageLabel_;
    QLineEdit* questionEdit_;
    QLabel* answerLabel_;
    QPushButton* selectBtn_;
    QPushButton* askBtn_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
