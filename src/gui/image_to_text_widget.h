#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QListWidget>
#include <QString>
#include <opencv2/opencv.hpp>
#include "../core/model_manager.h"
#include "../index/text_corpus_index.h"

namespace vindex {
namespace gui {

/**
 * @brief 图搜文界面（使用内置示例语料）
 */
class ImageToTextWidget : public QWidget {
    Q_OBJECT
public:
    ImageToTextWidget(core::ModelManager* modelManager,
                      QWidget* parent = nullptr);
    ~ImageToTextWidget() = default;

private slots:
    void onSelectImage();
    void onSearch();

private:
    void setupUI();
    void loadCorpus();
    void showError(const QString& message);
    void displayImage(const QString& path);

private:
    core::ModelManager* modelManager_;
    index::TextCorpusIndex corpus_;
    bool corpusReady_;

    QLabel* imageLabel_;
    QPushButton* selectBtn_;
    QPushButton* searchBtn_;
    QSpinBox* topKSpinBox_;
    QLineEdit* thresholdEdit_;
    QLabel* statusLabel_;
    QListWidget* resultList_;

    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex

