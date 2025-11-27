#include "main_window.h"
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QLabel>
#include <QSettings>
#include <QCloseEvent>
#include <QApplication>
#include <QDesktopServices>
#include <QUrl>
#include <QProgressDialog>
#include <QFile>
#include <filesystem>
#include "text_search_widget.h"
#include "match_widget.h"
#include "image_to_text_widget.h"
#include "api_ai_widget.h"
#include "caption_widget.h"
#include "vqa_widget.h"
#include "ocr_widget.h"
#include "database_widget.h"

namespace fs = std::filesystem;

using vindex::utils::Translator;

namespace vindex {
namespace gui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , modelManager_(&core::ModelManager::instance())
{
    setWindowTitle(TR("VIndex - Visual Search Engine"));
    setMinimumSize(1200, 800);

    // 加载样式表
    loadStyleSheet();

    // 加载设置（包括语言设置）
    loadSettings();

    // 初始化UI
    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();

    // 连接语言切换信号
    connect(&Translator::instance(), &Translator::languageChanged,
            this, &MainWindow::onLanguageChanged);

    // 初始化后端
    loadModels();
    initializeDatabase();
}

MainWindow::~MainWindow() {
    saveSettings();
}

void MainWindow::setupUI() {
    // 创建中央标签页部件
    tabWidget_ = new QTabWidget(this);
    tabWidget_->setTabPosition(QTabWidget::North);
    setCentralWidget(tabWidget_);

    // 注意：图搜图标签页将在数据库初始化后创建
}

void MainWindow::setupMenuBar() {
    // 文件菜单
    fileMenu_ = menuBar()->addMenu(TR("&File"));

    importAction_ = new QAction(TR("&Import Folder..."), this);
    importAction_->setShortcut(QKeySequence("Ctrl+I"));
    connect(importAction_, &QAction::triggered, this, &MainWindow::onImportFolder);
    fileMenu_->addAction(importAction_);

    fileMenu_->addSeparator();

    exitAction_ = new QAction(TR("E&xit"), this);
    exitAction_->setShortcut(QKeySequence("Ctrl+Q"));
    connect(exitAction_, &QAction::triggered, this, &QMainWindow::close);
    fileMenu_->addAction(exitAction_);

    // 数据库菜单
    databaseMenu_ = menuBar()->addMenu(TR("&Database"));

    rebuildAction_ = new QAction(TR("&Rebuild Index"), this);
    connect(rebuildAction_, &QAction::triggered, this, &MainWindow::onRebuildIndex);
    databaseMenu_->addAction(rebuildAction_);

    statsAction_ = new QAction(TR("&Statistics"), this);
    connect(statsAction_, &QAction::triggered, this, &MainWindow::onDatabaseStats);
    databaseMenu_->addAction(statsAction_);

    // 设置菜单
    settingsMenu_ = menuBar()->addMenu(TR("&Settings"));

    preferencesAction_ = new QAction(TR("&Preferences..."), this);
    connect(preferencesAction_, &QAction::triggered, this, &MainWindow::onSettings);
    settingsMenu_->addAction(preferencesAction_);

    settingsMenu_->addSeparator();

    // 语言子菜单
    languageMenu_ = settingsMenu_->addMenu(TR("&Language"));

    QActionGroup* langGroup = new QActionGroup(this);
    langGroup->setExclusive(true);

    englishAction_ = new QAction("English", this);
    englishAction_->setCheckable(true);
    englishAction_->setChecked(Translator::instance().currentLanguage() == Translator::English);
    connect(englishAction_, &QAction::triggered, this, &MainWindow::onSwitchToEnglish);
    langGroup->addAction(englishAction_);
    languageMenu_->addAction(englishAction_);

    chineseAction_ = new QAction("中文", this);
    chineseAction_->setCheckable(true);
    chineseAction_->setChecked(Translator::instance().currentLanguage() == Translator::Chinese);
    connect(chineseAction_, &QAction::triggered, this, &MainWindow::onSwitchToChinese);
    langGroup->addAction(chineseAction_);
    languageMenu_->addAction(chineseAction_);

    // 帮助菜单
    helpMenu_ = menuBar()->addMenu(TR("&Help"));

    aboutAction_ = new QAction(TR("&About"), this);
    connect(aboutAction_, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu_->addAction(aboutAction_);
}

void MainWindow::setupToolBar() {
    toolbar_ = addToolBar(TR("Main Toolbar"));
    toolbar_->setMovable(false);

    toolbarImportAction_ = new QAction(TR("Import Folder"), this);
    connect(toolbarImportAction_, &QAction::triggered, this, &MainWindow::onImportFolder);
    toolbar_->addAction(toolbarImportAction_);

    toolbar_->addSeparator();

    toolbarRebuildAction_ = new QAction(TR("Rebuild Index"), this);
    connect(toolbarRebuildAction_, &QAction::triggered, this, &MainWindow::onRebuildIndex);
    toolbar_->addAction(toolbarRebuildAction_);
}

void MainWindow::setupStatusBar() {
    statusLabel_ = new QLabel(TR("Ready"), this);
    statusBar()->addWidget(statusLabel_);

    dbStatsLabel_ = new QLabel(TR("Images: %1").arg(0), this);
    statusBar()->addPermanentWidget(dbStatsLabel_);
}

void MainWindow::onSwitchToEnglish() {
    Translator::instance().setLanguage(Translator::English);
}

void MainWindow::onSwitchToChinese() {
    Translator::instance().setLanguage(Translator::Chinese);
}

void MainWindow::onLanguageChanged() {
    retranslateUI();
}

void MainWindow::retranslateUI() {
    // 更新窗口标题
    setWindowTitle(TR("VIndex - Visual Search Engine"));

    // 更新菜单
    fileMenu_->setTitle(TR("&File"));
    importAction_->setText(TR("&Import Folder..."));
    exitAction_->setText(TR("E&xit"));

    databaseMenu_->setTitle(TR("&Database"));
    rebuildAction_->setText(TR("&Rebuild Index"));
    statsAction_->setText(TR("&Statistics"));

    settingsMenu_->setTitle(TR("&Settings"));
    preferencesAction_->setText(TR("&Preferences..."));
    languageMenu_->setTitle(TR("&Language"));

    helpMenu_->setTitle(TR("&Help"));
    aboutAction_->setText(TR("&About"));

    // 更新工具栏
    toolbar_->setWindowTitle(TR("Main Toolbar"));
    toolbarImportAction_->setText(TR("Import Folder"));
    toolbarRebuildAction_->setText(TR("Rebuild Index"));

    // 更新状态栏
    statusLabel_->setText(TR("Ready"));
    if (dbManager_) {
        int64_t imageCount = dbManager_->totalCount();
        dbStatsLabel_->setText(TR("Images: %1").arg(imageCount));
    }

    // 更新标签页名称
    if (tabWidget_->count() >= 8) {
        tabWidget_->setTabText(0, TR("Image Search"));
        tabWidget_->setTabText(1, TR("Text Search"));
        tabWidget_->setTabText(2, TR("Image→Text"));
        tabWidget_->setTabText(3, TR("API AI"));
        tabWidget_->setTabText(4, TR("Match"));
        tabWidget_->setTabText(5, TR("Caption"));
        tabWidget_->setTabText(6, TR("VQA"));
        tabWidget_->setTabText(7, TR("Library"));
    }

    // 更新语言选择状态
    englishAction_->setChecked(Translator::instance().currentLanguage() == Translator::English);
    chineseAction_->setChecked(Translator::instance().currentLanguage() == Translator::Chinese);
}

void MainWindow::loadModels() {
    QProgressDialog loadingDialog(TR("Loading models..."), QString(), 0, 0, this);
    loadingDialog.setWindowModality(Qt::WindowModal);
    loadingDialog.setCancelButton(nullptr);
    loadingDialog.setMinimumDuration(0);
    loadingDialog.show();

    QApplication::processEvents();

    try {
        // 配置模型路径
        std::string modelPath = "./assets/models";
        std::string vocabPath = "./assets/vocab/clip_vocab.txt";

        // 检查路径是否存在
        if (!fs::exists(modelPath)) {
            QMessageBox::warning(
                this,
                TR("Warning"),
                TR("Model directory not found. Please ensure models are in ./assets/models/\n\nRun the Python export script first:\n  cd scripts && python export_clip_to_onnx.py")
            );
        }

        modelManager_->setModelPath(modelPath);
        modelManager_->setVocabPath(vocabPath);
        modelManager_->setEmbeddingDim(512);  // CN-CLIP 默认512维

        statusLabel_->setText(TR("Models configured successfully"));

    } catch (const std::exception& e) {
        QMessageBox::critical(
            this,
            TR("Error"),
            TR("Failed to load models: %1").arg(e.what())
        );
    }

    loadingDialog.close();
}

void MainWindow::initializeDatabase() {
    try {
        // 创建数据目录
        std::string dataDir = "./data";
        fs::create_directories(dataDir);

        // 初始化数据库
        std::string dbPath = dataDir + "/vindex.db";
        std::string indexPath = dataDir + "/index/faiss.index";

        fs::create_directories(dataDir + "/index");

        dbManager_ = std::make_unique<index::DatabaseManager>(dbPath, indexPath, modelManager_->getEmbeddingDim());

        if (!dbManager_->initialize()) {
            throw std::runtime_error("Failed to initialize database");
        }

        // 设置编码器（懒加载）
        dbManager_->setEncoder(&modelManager_->clipEncoder());

        // 创建图搜图标签页
        imageSearchTab_ = new ImageSearchWidget(dbManager_.get(), this);
        tabWidget_->addTab(imageSearchTab_, TR("Image Search"));
        // 文搜图
        textSearchTab_ = new TextSearchWidget(dbManager_.get(), this);
        tabWidget_->addTab(textSearchTab_, TR("Text Search"));
        // 图搜文（示例语料）
        imageToTextTab_ = new ImageToTextWidget(modelManager_, this);
        tabWidget_->addTab(imageToTextTab_, TR("Image→Text"));
        // 远程 API（文生图 / 图生文 VQA）
        apiTab_ = new ApiAIWidget(modelManager_, this);
        tabWidget_->addTab(apiTab_, TR("API AI"));
        // 图文匹配
        matchTab_ = new MatchWidget(modelManager_, this);
        tabWidget_->addTab(matchTab_, TR("Match"));
        // 图生文
        captionTab_ = new CaptionWidget(modelManager_, this);
        tabWidget_->addTab(captionTab_, TR("Caption"));
        // VQA
        vqaTab_ = new VQAWidget(modelManager_, this);
        tabWidget_->addTab(vqaTab_, TR("VQA"));
        // OCR
        ocrTab_ = new OcrWidget(modelManager_, this);
        tabWidget_->addTab(ocrTab_, TR("OCR"));
        // 图库管理
        databaseTab_ = new DatabaseWidget(dbManager_.get(), this);
        tabWidget_->addTab(databaseTab_, TR("Library"));

        // 更新统计信息
        int64_t imageCount = dbManager_->totalCount();
        dbStatsLabel_->setText(TR("Images: %1").arg(imageCount));

        statusLabel_->setText(TR("Database initialized successfully"));

    } catch (const std::exception& e) {
        QMessageBox::critical(
            this,
            TR("Error"),
            TR("Failed to initialize database: %1").arg(e.what())
        );
    }
}

void MainWindow::onImportFolder() {
    QString folderPath = QFileDialog::getExistingDirectory(
        this,
        TR("Select Image Folder"),
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    if (folderPath.isEmpty()) {
        return;
    }

    // 询问是否递归
    auto reply = QMessageBox::question(
        this,
        TR("Import Options"),
        TR("Include subdirectories?"),
        QMessageBox::Yes | QMessageBox::No
    );

    bool recursive = (reply == QMessageBox::Yes);

    // 创建进度对话框
    QProgressDialog progress(TR("Importing images..."), TR("Cancel"), 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    try {
        size_t importedCount = dbManager_->importFolder(
            folderPath.toStdString(),
            recursive,
            [&progress](int current, int total) {
                progress.setMaximum(total);
                progress.setValue(current);
                QApplication::processEvents();
            }
        );

        progress.close();

        QMessageBox::information(
            this,
            TR("Import Complete"),
            TR("Successfully imported %1 images").arg(importedCount)
        );

        // 更新统计信息
        int64_t imageCount = dbManager_->totalCount();
        dbStatsLabel_->setText(TR("Images: %1").arg(imageCount));

        // 保存索引
        dbManager_->saveIndex();

    } catch (const std::exception& e) {
        progress.close();
        QMessageBox::critical(
            this,
            TR("Error"),
            TR("Import failed: %1").arg(e.what())
        );
    }
}

void MainWindow::onRebuildIndex() {
    auto reply = QMessageBox::question(
        this,
        TR("Rebuild Index"),
        TR("This will rebuild the entire search index.\nThis may take a while depending on the number of images.\n\nContinue?"),
        QMessageBox::Yes | QMessageBox::No
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    QProgressDialog progress(TR("Rebuilding index..."), TR("Cancel"), 0, 100, this);
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    try {
        bool success = dbManager_->rebuildIndex(
            [&progress](int current, int total) {
                progress.setMaximum(total);
                progress.setValue(current);
                QApplication::processEvents();
            }
        );

        progress.close();

        if (success) {
            QMessageBox::information(
                this,
                TR("Success"),
                TR("Index rebuilt successfully")
            );
        } else {
            QMessageBox::warning(
                this,
                TR("Warning"),
                TR("Index rebuild completed with errors")
            );
        }

    } catch (const std::exception& e) {
        progress.close();
        QMessageBox::critical(
            this,
            TR("Error"),
            TR("Rebuild failed: %1").arg(e.what())
        );
    }
}

void MainWindow::onDatabaseStats() {
    int64_t totalCount = dbManager_->totalCount();
    size_t indexSize = dbManager_->faissIndex().size();
    auto categories = dbManager_->getAllCategories();

    QString stats = QString(
        "%1\n"
        "==================\n\n"
        "%2\n"
        "%3\n"
        "%4\n\n"
        "%5\n"
        "%6"
    ).arg(TR("Database Statistics"))
     .arg(TR("Total Images: %1").arg(totalCount))
     .arg(TR("Index Size: %1").arg(indexSize))
     .arg(TR("Categories: %1").arg(categories.size()))
     .arg(TR("Database Path: %1").arg(QString::fromStdString(dbManager_->getDbPath())))
     .arg(TR("Index Path: %1").arg(QString::fromStdString(dbManager_->getIndexPath())));

    QMessageBox::information(this, TR("Database Statistics"), stats);
}

void MainWindow::onSettings() {
    QMessageBox::information(
        this,
        TR("Settings"),
        TR("Settings dialog not yet implemented.\n\nConfigure model paths in code or via config file.")
    );
}

void MainWindow::onAbout() {
    QString aboutText;
    if (Translator::instance().currentLanguage() == Translator::Chinese) {
        aboutText =
            "<h2>VIndex - 视觉搜索引擎</h2>"
            "<p>版本 1.0.0</p>"
            "<p>一个强大的图像搜索应用，使用 CLIP 嵌入和 FAISS 索引。</p>"
            "<p><b>功能特性：</b></p>"
            "<ul>"
            "<li>以图搜图</li>"
            "<li>以文搜图</li>"
            "<li>基于 FAISS 的快速相似度搜索</li>"
            "<li>ONNX Runtime 推理引擎</li>"
            "</ul>"
            "<p>基于 Qt6、OpenCV、ONNX Runtime 和 FAISS 构建。</p>";
    } else {
        aboutText =
            "<h2>VIndex - Visual Search Engine</h2>"
            "<p>Version 1.0.0</p>"
            "<p>A powerful image search application using CLIP embeddings and FAISS indexing.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Image-to-image search</li>"
            "<li>Text-to-image search</li>"
            "<li>Fast similarity search with FAISS</li>"
            "<li>ONNX Runtime inference</li>"
            "</ul>"
            "<p>Built with Qt6, OpenCV, ONNX Runtime, and FAISS.</p>";
    }
    QMessageBox::about(this, TR("About VIndex"), aboutText);
}

void MainWindow::saveSettings() {
    QSettings settings("VIndex", "ImageSearch");

    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
    settings.setValue("language", static_cast<int>(Translator::instance().currentLanguage()));
}

void MainWindow::loadSettings() {
    QSettings settings("VIndex", "ImageSearch");

    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());

    // 加载语言设置
    int lang = settings.value("language", static_cast<int>(Translator::English)).toInt();
    Translator::instance().setLanguage(static_cast<Translator::Language>(lang));
}

void MainWindow::closeEvent(QCloseEvent* event) {
    saveSettings();

    // 保存索引
    if (dbManager_) {
        statusLabel_->setText(TR("Saving index..."));
        dbManager_->saveIndex();
    }

    event->accept();
}

void MainWindow::loadStyleSheet() {
    // 尝试多个可能的样式表路径
    QStringList stylePaths = {
        "./resources/styles/modern.qss",
        "../resources/styles/modern.qss",
        ":/styles/modern.qss"
    };

    for (const QString& path : stylePaths) {
        QFile styleFile(path);
        if (styleFile.open(QFile::ReadOnly | QFile::Text)) {
            QString styleSheet = QString::fromUtf8(styleFile.readAll());
            qApp->setStyleSheet(styleSheet);
            styleFile.close();
            qDebug() << "Loaded style sheet from:" << path;
            return;
        }
    }

    // 如果找不到外部样式表，使用内置基础样式
    QString fallbackStyle = R"(
        * {
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
            font-size: 13px;
        }
        QMainWindow {
            background-color: #f5f7fa;
        }
        QPushButton {
            background-color: #409eff;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            color: white;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #66b1ff;
        }
        QPushButton:pressed {
            background-color: #3a8ee6;
        }
        QPushButton:disabled {
            background-color: #a0cfff;
        }
        QLineEdit, QTextEdit, QSpinBox {
            background-color: white;
            border: 1px solid #dcdfe6;
            border-radius: 4px;
            padding: 6px 10px;
        }
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {
            border-color: #409eff;
        }
        QGroupBox {
            background-color: white;
            border: 1px solid #e4e7ed;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 24px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 4px;
        }
        QTabWidget::pane {
            background-color: white;
            border: 1px solid #e4e7ed;
            border-radius: 8px;
        }
        QTabBar::tab {
            background-color: #f5f7fa;
            border: 1px solid #e4e7ed;
            padding: 8px 20px;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background-color: white;
            color: #409eff;
        }
    )";
    qApp->setStyleSheet(fallbackStyle);
}

} // namespace gui
} // namespace vindex
