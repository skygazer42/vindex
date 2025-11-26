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
#include <filesystem>
#include "text_search_widget.h"
#include "match_widget.h"
#include "caption_widget.h"
#include "vqa_widget.h"
#include "database_widget.h"

namespace fs = std::filesystem;

namespace vindex {
namespace gui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , modelManager_(&core::ModelManager::instance())
    , loadingDialog_(nullptr)
{
    setWindowTitle("VIndex - Visual Search Engine");
    setMinimumSize(1200, 800);

    // 加载设置
    loadSettings();

    // 初始化UI
    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();

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
    QMenu* fileMenu = menuBar()->addMenu("&File");

    QAction* importAction = new QAction("&Import Folder...", this);
    importAction->setShortcut(QKeySequence("Ctrl+I"));
    connect(importAction, &QAction::triggered, this, &MainWindow::onImportFolder);
    fileMenu->addAction(importAction);

    fileMenu->addSeparator();

    QAction* exitAction = new QAction("E&xit", this);
    exitAction->setShortcut(QKeySequence("Ctrl+Q"));
    connect(exitAction, &QAction::triggered, this, &QMainWindow::close);
    fileMenu->addAction(exitAction);

    // 数据库菜单
    QMenu* databaseMenu = menuBar()->addMenu("&Database");

    QAction* rebuildAction = new QAction("&Rebuild Index", this);
    connect(rebuildAction, &QAction::triggered, this, &MainWindow::onRebuildIndex);
    databaseMenu->addAction(rebuildAction);

    QAction* statsAction = new QAction("&Statistics", this);
    connect(statsAction, &QAction::triggered, this, &MainWindow::onDatabaseStats);
    databaseMenu->addAction(statsAction);

    // 设置菜单
    QMenu* settingsMenu = menuBar()->addMenu("&Settings");

    QAction* preferencesAction = new QAction("&Preferences...", this);
    connect(preferencesAction, &QAction::triggered, this, &MainWindow::onSettings);
    settingsMenu->addAction(preferencesAction);

    // 帮助菜单
    QMenu* helpMenu = menuBar()->addMenu("&Help");

    QAction* aboutAction = new QAction("&About", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::onAbout);
    helpMenu->addAction(aboutAction);
}

void MainWindow::setupToolBar() {
    QToolBar* toolbar = addToolBar("Main Toolbar");
    toolbar->setMovable(false);

    QAction* importAction = new QAction("Import Folder", this);
    connect(importAction, &QAction::triggered, this, &MainWindow::onImportFolder);
    toolbar->addAction(importAction);

    toolbar->addSeparator();

    QAction* rebuildAction = new QAction("Rebuild Index", this);
    connect(rebuildAction, &QAction::triggered, this, &MainWindow::onRebuildIndex);
    toolbar->addAction(rebuildAction);
}

void MainWindow::setupStatusBar() {
    statusLabel_ = new QLabel("Ready", this);
    statusBar()->addWidget(statusLabel_);

    dbStatsLabel_ = new QLabel("Images: 0", this);
    statusBar()->addPermanentWidget(dbStatsLabel_);
}

void MainWindow::loadModels() {
    loadingDialog_ = new QProgressDialog("Loading models...", QString(), 0, 0, this);
    loadingDialog_->setWindowModality(Qt::WindowModal);
    loadingDialog_->setCancelButton(nullptr);
    loadingDialog_->setMinimumDuration(0);
    loadingDialog_->show();

    QApplication::processEvents();

    try {
        // 配置模型路径
        std::string modelPath = "./assets/models";
        std::string vocabPath = "./assets/vocab/bpe_simple_vocab_16e6.txt";

        // 检查路径是否存在
        if (!fs::exists(modelPath)) {
            QMessageBox::warning(
                this,
                "Warning",
                "Model directory not found. Please ensure models are in ./assets/models/\n\n"
                "Run the Python export script first:\n"
                "  cd scripts && python export_clip_to_onnx.py"
            );
        }

        modelManager_->setModelPath(modelPath);
        modelManager_->setVocabPath(vocabPath);
        modelManager_->setEmbeddingDim(768);  // ViT-L/14

        // 预加载模型（可选）
        // modelManager_->preloadAll();

        statusLabel_->setText("Models configured successfully");

    } catch (const std::exception& e) {
        QMessageBox::critical(
            this,
            "Error",
            QString("Failed to load models: %1").arg(e.what())
        );
    }

    loadingDialog_->close();
    delete loadingDialog_;
    loadingDialog_ = nullptr;
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

        dbManager_ = std::make_unique<index::DatabaseManager>(dbPath, indexPath, 768);

        if (!dbManager_->initialize()) {
            throw std::runtime_error("Failed to initialize database");
        }

        // 设置编码器（懒加载）
        dbManager_->setEncoder(&modelManager_->clipEncoder());

        // 创建图搜图标签页
        imageSearchTab_ = new ImageSearchWidget(dbManager_.get(), this);
        tabWidget_->addTab(imageSearchTab_, "Image Search");
        // 文搜图
        textSearchTab_ = new TextSearchWidget(dbManager_.get(), this);
        tabWidget_->addTab(textSearchTab_, "Text Search");
        // 图文匹配
        matchTab_ = new MatchWidget(modelManager_, this);
        tabWidget_->addTab(matchTab_, "Match");
        // 图生文
        captionTab_ = new CaptionWidget(modelManager_, this);
        tabWidget_->addTab(captionTab_, "Caption");
        // VQA
        vqaTab_ = new VQAWidget(modelManager_, this);
        tabWidget_->addTab(vqaTab_, "VQA");
        // 图库管理
        databaseTab_ = new DatabaseWidget(dbManager_.get(), this);
        tabWidget_->addTab(databaseTab_, "Library");

        // 更新统计信息
        int64_t imageCount = dbManager_->totalCount();
        dbStatsLabel_->setText(QString("Images: %1").arg(imageCount));

        statusLabel_->setText("Database initialized successfully");

    } catch (const std::exception& e) {
        QMessageBox::critical(
            this,
            "Error",
            QString("Failed to initialize database: %1").arg(e.what())
        );
    }
}

void MainWindow::onImportFolder() {
    QString folderPath = QFileDialog::getExistingDirectory(
        this,
        "Select Image Folder",
        QString(),
        QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks
    );

    if (folderPath.isEmpty()) {
        return;
    }

    // 询问是否递归
    auto reply = QMessageBox::question(
        this,
        "Import Options",
        "Include subdirectories?",
        QMessageBox::Yes | QMessageBox::No
    );

    bool recursive = (reply == QMessageBox::Yes);

    // 创建进度对话框
    QProgressDialog progress("Importing images...", "Cancel", 0, 100, this);
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
            "Import Complete",
            QString("Successfully imported %1 images").arg(importedCount)
        );

        // 更新统计信息
        int64_t imageCount = dbManager_->totalCount();
        dbStatsLabel_->setText(QString("Images: %1").arg(imageCount));

        // 保存索引
        dbManager_->saveIndex();

    } catch (const std::exception& e) {
        progress.close();
        QMessageBox::critical(
            this,
            "Error",
            QString("Import failed: %1").arg(e.what())
        );
    }
}

void MainWindow::onRebuildIndex() {
    auto reply = QMessageBox::question(
        this,
        "Rebuild Index",
        "This will rebuild the entire search index.\n"
        "This may take a while depending on the number of images.\n\n"
        "Continue?",
        QMessageBox::Yes | QMessageBox::No
    );

    if (reply != QMessageBox::Yes) {
        return;
    }

    QProgressDialog progress("Rebuilding index...", "Cancel", 0, 100, this);
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
                "Success",
                "Index rebuilt successfully"
            );
        } else {
            QMessageBox::warning(
                this,
                "Warning",
                "Index rebuild completed with errors"
            );
        }

    } catch (const std::exception& e) {
        progress.close();
        QMessageBox::critical(
            this,
            "Error",
            QString("Rebuild failed: %1").arg(e.what())
        );
    }
}

void MainWindow::onDatabaseStats() {
    int64_t totalCount = dbManager_->totalCount();
    size_t indexSize = dbManager_->faissIndex().size();
    auto categories = dbManager_->getAllCategories();

    QString stats = QString(
        "Database Statistics\n"
        "==================\n\n"
        "Total Images: %1\n"
        "Index Size: %2\n"
        "Categories: %3\n\n"
        "Database Path: %4\n"
        "Index Path: %5"
    ).arg(totalCount)
     .arg(indexSize)
     .arg(categories.size())
     .arg(QString::fromStdString(dbManager_->getDbPath()))
     .arg(QString::fromStdString(dbManager_->getIndexPath()));

    QMessageBox::information(this, "Database Statistics", stats);
}

void MainWindow::onSettings() {
    QMessageBox::information(
        this,
        "Settings",
        "Settings dialog not yet implemented.\n\n"
        "Configure model paths in code or via config file."
    );
}

void MainWindow::onAbout() {
    QMessageBox::about(
        this,
        "About VIndex",
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
        "<p>Built with Qt6, OpenCV, ONNX Runtime, and FAISS.</p>"
    );
}

void MainWindow::saveSettings() {
    QSettings settings("VIndex", "ImageSearch");

    settings.setValue("geometry", saveGeometry());
    settings.setValue("windowState", saveState());
}

void MainWindow::loadSettings() {
    QSettings settings("VIndex", "ImageSearch");

    restoreGeometry(settings.value("geometry").toByteArray());
    restoreState(settings.value("windowState").toByteArray());
}

void MainWindow::closeEvent(QCloseEvent* event) {
    saveSettings();

    // 保存索引
    if (dbManager_) {
        statusLabel_->setText("Saving index...");
        dbManager_->saveIndex();
    }

    event->accept();
}

} // namespace gui
} // namespace vindex
