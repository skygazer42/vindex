#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QProgressBar>
#include <QString>
#include "image_gallery.h"
#include "../index/database_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 文搜图界面
 *
 * 提供文本查询输入，调用数据库/索引进行检索。
 * 若未加载文本模型，会给出友好提示。
 */
class TextSearchWidget : public QWidget {
    Q_OBJECT

public:
    explicit TextSearchWidget(index::DatabaseManager* dbManager,
                              QWidget* parent = nullptr);
    ~TextSearchWidget() = default;

signals:
    void searchCompleted(int resultCount);
    void errorOccurred(const QString& message);

private slots:
    void onSearch();
    void onResultClicked(int64_t imageId);
    void onResultDoubleClicked(int64_t imageId);

private:
    void setupUI();
    void performSearch();
    void showError(const QString& message);

private:
    index::DatabaseManager* dbManager_;

    QLineEdit* queryEdit_;
    QSpinBox* topKSpinBox_;
    QLineEdit* thresholdEdit_;
    QPushButton* searchBtn_;
    QLabel* statusLabel_;
    QProgressBar* progressBar_;
    ImageGallery* resultGallery_;
};

} // namespace gui
} // namespace vindex
