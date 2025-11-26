#pragma once

#include <QWidget>
#include <QLabel>
#include <QLineEdit>
#include <QTextEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QProgressBar>
#include <QListWidget>
#include <QString>
#include <QStringList>
#include "image_gallery.h"
#include "../index/database_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 文搜图界面
 *
 * 功能：
 * - 文本查询输入（支持多行）
 * - 搜索参数配置（Top-K、阈值）
 * - 搜索历史记录
 * - 示例查询
 * - 结果展示
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
    void onClear();
    void onHistoryClicked(QListWidgetItem* item);
    void onExampleClicked();
    void onResultClicked(int64_t imageId);
    void onResultDoubleClicked(int64_t imageId);
    void onQueryTextChanged();

private:
    void setupUI();
    void createExamples();
    void performSearch(const QString& queryText);
    void showError(const QString& message);
    void addToHistory(const QString& query, int resultCount);
    void loadHistory();
    void saveHistory();

private:
    // 数据库管理器
    index::DatabaseManager* dbManager_;

    // UI组件 - 查询区域
    QTextEdit* queryTextEdit_;          // 多行文本输入
    QPushButton* searchBtn_;            // 搜索按钮
    QPushButton* clearBtn_;             // 清空按钮
    QSpinBox* topKSpinBox_;             // Top-K选择器
    QLineEdit* thresholdEdit_;          // 相似度阈值

    // UI组件 - 历史和示例
    QListWidget* historyList_;          // 搜索历史
    QPushButton* clearHistoryBtn_;      // 清空历史
    QWidget* examplesWidget_;           // 示例查询区域

    // UI组件 - 状态和结果
    QLabel* statusLabel_;               // 状态标签
    QProgressBar* progressBar_;         // 进度条
    ImageGallery* resultGallery_;       // 结果展示

    // 数据
    QStringList searchHistory_;         // 搜索历史
    static const int MAX_HISTORY = 20;  // 最大历史数
};

} // namespace gui
} // namespace vindex
