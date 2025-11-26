#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSpinBox>
#include <QLineEdit>
#include <QComboBox>
#include <QProgressBar>
#include <QString>
#include <opencv2/opencv.hpp>
#include "image_gallery.h"
#include "../index/database_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 图搜图界面
 *
 * 功能：
 * - 选择查询图像
 * - 设置搜索参数（Top-K、相似度阈值）
 * - 显示查询图像
 * - 展示搜索结果
 */
class ImageSearchWidget : public QWidget {
    Q_OBJECT

public:
    explicit ImageSearchWidget(index::DatabaseManager* dbManager,
                              QWidget* parent = nullptr);
    ~ImageSearchWidget() = default;

signals:
    void searchCompleted(int resultCount);
    void errorOccurred(const QString& message);

private slots:
    void onSelectImage();
    void onSearch();
    void onResultClicked(int64_t imageId);
    void onResultDoubleClicked(int64_t imageId);

private:
    void setupUI();
    void displayQueryImage(const QString& imagePath);
    void performSearch();
    void showError(const QString& message);

private:
    // 数据库管理器
    index::DatabaseManager* dbManager_;

    // UI组件
    QLabel* queryImageLabel_;         // 查询图像显示
    QPushButton* selectImageBtn_;     // 选择图像按钮
    QPushButton* searchBtn_;          // 搜索按钮
    QSpinBox* topKSpinBox_;           // Top-K选择器
    QLineEdit* thresholdEdit_;        // 相似度阈值输入
    QLabel* statusLabel_;             // 状态标签
    QProgressBar* progressBar_;       // 进度条
    ImageGallery* resultGallery_;     // 结果展示

    // 当前状态
    QString currentQueryPath_;        // 当前查询图像路径
};

} // namespace gui
} // namespace vindex
