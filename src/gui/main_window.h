#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QProgressDialog>
#include <memory>

#include "image_search_widget.h"
#include "../index/database_manager.h"
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 主窗口
 *
 * 包含：
 * - 菜单栏
 * - 工具栏
 * - 标签页（图搜图、文搜图、图库管理等）
 * - 状态栏
 */
class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void onImportFolder();
    void onRebuildIndex();
    void onAbout();
    void onSettings();
    void onDatabaseStats();

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void loadModels();
    void initializeDatabase();
    void saveSettings();
    void loadSettings();

private:
    // 核心组件
    std::unique_ptr<index::DatabaseManager> dbManager_;
    core::ModelManager* modelManager_;

    // UI组件
    QTabWidget* tabWidget_;
    ImageSearchWidget* imageSearchTab_;

    // 对话框
    QProgressDialog* loadingDialog_;

    // 状态栏
    QLabel* statusLabel_;
    QLabel* dbStatsLabel_;
};

} // namespace gui
} // namespace vindex
