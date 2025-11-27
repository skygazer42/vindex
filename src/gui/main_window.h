#pragma once

#include <QMainWindow>
#include <QTabWidget>
#include <QMenuBar>
#include <QToolBar>
#include <QStatusBar>
#include <QActionGroup>
#include <memory>

#include "image_search_widget.h"
#include "../index/database_manager.h"
#include "../core/model_manager.h"
#include "../utils/translator.h"

namespace vindex {
namespace gui {

class TextSearchWidget;
class MatchWidget;
class CaptionWidget;
class VQAWidget;
class OcrWidget;
class DatabaseWidget;
class ImageToTextWidget;
class ApiAIWidget;

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
    void onLanguageChanged();
    void onSwitchToEnglish();
    void onSwitchToChinese();

private:
    void setupUI();
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void loadModels();
    void initializeDatabase();
    void saveSettings();
    void loadSettings();
    void loadStyleSheet();
    void retranslateUI();

private:
    // 核心组件
    std::unique_ptr<index::DatabaseManager> dbManager_;
    core::ModelManager* modelManager_;

    // UI组件
    QTabWidget* tabWidget_;
    ImageSearchWidget* imageSearchTab_;
    class TextSearchWidget* textSearchTab_;
    class MatchWidget* matchTab_;
    class ImageToTextWidget* imageToTextTab_;
    class ApiAIWidget* apiTab_;
    class CaptionWidget* captionTab_;
    class VQAWidget* vqaTab_;
    class OcrWidget* ocrTab_;
    class DatabaseWidget* databaseTab_;

    // 状态栏
    QLabel* statusLabel_;
    QLabel* dbStatsLabel_;

    // 菜单项（需要保存引用以便更新文本）
    QMenu* fileMenu_;
    QMenu* databaseMenu_;
    QMenu* settingsMenu_;
    QMenu* languageMenu_;
    QMenu* helpMenu_;
    QAction* importAction_;
    QAction* exitAction_;
    QAction* rebuildAction_;
    QAction* statsAction_;
    QAction* preferencesAction_;
    QAction* englishAction_;
    QAction* chineseAction_;
    QAction* aboutAction_;
    QToolBar* toolbar_;
    QAction* toolbarImportAction_;
    QAction* toolbarRebuildAction_;
};

} // namespace gui
} // namespace vindex
