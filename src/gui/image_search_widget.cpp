#include "image_search_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QImageReader>
#include <QLabel>
#include <QFileInfo>
#include <QDesktopServices>
#include <QUrl>

namespace vindex {
namespace gui {

ImageSearchWidget::ImageSearchWidget(index::DatabaseManager* dbManager,
                                    QWidget* parent)
    : QWidget(parent)
    , dbManager_(dbManager)
{
    setupUI();
}

void ImageSearchWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // ==================== 查询区域 ====================
    auto* queryGroup = new QGroupBox("Query Image", this);
    auto* queryLayout = new QHBoxLayout(queryGroup);

    // 查询图像显示
    queryImageLabel_ = new QLabel(this);
    queryImageLabel_->setFixedSize(300, 300);
    queryImageLabel_->setAlignment(Qt::AlignCenter);
    queryImageLabel_->setScaledContents(true);
    queryImageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    queryImageLabel_->setText("No image selected\n\nClick 'Select Image' to choose");
    queryLayout->addWidget(queryImageLabel_);

    // 控制面板
    auto* controlPanel = new QWidget(this);
    auto* controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setSpacing(10);

    // 选择图像按钮
    selectImageBtn_ = new QPushButton("Select Image", this);
    selectImageBtn_->setMinimumHeight(40);
    selectImageBtn_->setStyleSheet(
        "QPushButton { background-color: #0066cc; color: white; "
        "font-size: 14px; border-radius: 5px; }"
        "QPushButton:hover { background-color: #0052a3; }"
    );
    connect(selectImageBtn_, &QPushButton::clicked,
            this, &ImageSearchWidget::onSelectImage);
    controlLayout->addWidget(selectImageBtn_);

    // Top-K 设置
    auto* topKLayout = new QHBoxLayout();
    topKLayout->addWidget(new QLabel("Top K:", this));
    topKSpinBox_ = new QSpinBox(this);
    topKSpinBox_->setRange(1, 100);
    topKSpinBox_->setValue(10);
    topKSpinBox_->setMinimumWidth(80);
    topKLayout->addWidget(topKSpinBox_);
    topKLayout->addStretch();
    controlLayout->addLayout(topKLayout);

    // 相似度阈值
    auto* thresholdLayout = new QHBoxLayout();
    thresholdLayout->addWidget(new QLabel("Threshold:", this));
    thresholdEdit_ = new QLineEdit("0.0", this);
    thresholdEdit_->setPlaceholderText("0.0 - 1.0");
    thresholdEdit_->setMaximumWidth(80);
    thresholdLayout->addWidget(thresholdEdit_);
    thresholdLayout->addStretch();
    controlLayout->addLayout(thresholdLayout);

    // 搜索按钮
    searchBtn_ = new QPushButton("Search", this);
    searchBtn_->setMinimumHeight(40);
    searchBtn_->setEnabled(false);
    searchBtn_->setStyleSheet(
        "QPushButton { background-color: #28a745; color: white; "
        "font-size: 14px; font-weight: bold; border-radius: 5px; }"
        "QPushButton:hover { background-color: #218838; }"
        "QPushButton:disabled { background-color: #ccc; }"
    );
    connect(searchBtn_, &QPushButton::clicked,
            this, &ImageSearchWidget::onSearch);
    controlLayout->addWidget(searchBtn_);

    controlLayout->addStretch();

    queryLayout->addWidget(controlPanel);
    mainLayout->addWidget(queryGroup);

    // ==================== 状态栏 ====================
    auto* statusLayout = new QHBoxLayout();

    statusLabel_ = new QLabel("Ready", this);
    statusLabel_->setStyleSheet("QLabel { color: #666; }");
    statusLayout->addWidget(statusLabel_);

    progressBar_ = new QProgressBar(this);
    progressBar_->setVisible(false);
    progressBar_->setMaximumWidth(200);
    statusLayout->addWidget(progressBar_);

    statusLayout->addStretch();
    mainLayout->addLayout(statusLayout);

    // ==================== 结果展示区域 ====================
    auto* resultGroup = new QGroupBox("Search Results", this);
    auto* resultLayout = new QVBoxLayout(resultGroup);

    resultGallery_ = new ImageGallery(this);
    connect(resultGallery_, &ImageGallery::itemClicked,
            this, &ImageSearchWidget::onResultClicked);
    connect(resultGallery_, &ImageGallery::itemDoubleClicked,
            this, &ImageSearchWidget::onResultDoubleClicked);

    resultLayout->addWidget(resultGallery_);
    mainLayout->addWidget(resultGroup, 1);  // 占据剩余空间
}

void ImageSearchWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Query Image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    currentQueryPath_ = fileName;
    displayQueryImage(fileName);
    searchBtn_->setEnabled(true);
    statusLabel_->setText("Image loaded: " + QFileInfo(fileName).fileName());
}

void ImageSearchWidget::displayQueryImage(const QString& imagePath) {
    QImageReader reader(imagePath);
    reader.setScaledSize(QSize(300, 300));

    QImage image = reader.read();
    if (image.isNull()) {
        showError("Failed to load image: " + imagePath);
        return;
    }

    queryImageLabel_->setPixmap(QPixmap::fromImage(image));
}

void ImageSearchWidget::onSearch() {
    if (currentQueryPath_.isEmpty()) {
        showError("Please select a query image first");
        return;
    }

    if (!dbManager_) {
        showError("Database manager not initialized");
        return;
    }

    performSearch();
}

void ImageSearchWidget::performSearch() {
    // 禁用按钮
    searchBtn_->setEnabled(false);
    selectImageBtn_->setEnabled(false);
    statusLabel_->setText("Searching...");
    progressBar_->setVisible(true);
    progressBar_->setRange(0, 0);  // 不确定进度

    try {
        // 获取参数
        int topK = topKSpinBox_->value();
        float threshold = thresholdEdit_->text().toFloat();

        // 执行搜索
        auto results = dbManager_->searchByImage(
            currentQueryPath_.toStdString(),
            topK,
            threshold
        );

        // 构建结果
        std::vector<ImageGallery::GalleryItem> items;
        items.reserve(results.size());

        for (const auto& result : results) {
            QString label = QString("%1 (%2x%3)")
                .arg(QString::fromStdString(result.record.fileName))
                .arg(result.record.width)
                .arg(result.record.height);

            items.emplace_back(
                result.record.id,
                QString::fromStdString(result.record.filePath),
                result.score,
                label
            );
        }

        // 显示结果
        resultGallery_->setResults(items);

        statusLabel_->setText(QString("Found %1 results").arg(results.size()));
        emit searchCompleted(results.size());

    } catch (const std::exception& e) {
        showError(QString("Search failed: %1").arg(e.what()));
    }

    // 恢复按钮
    searchBtn_->setEnabled(true);
    selectImageBtn_->setEnabled(true);
    progressBar_->setVisible(false);
}

void ImageSearchWidget::onResultClicked(int64_t imageId) {
    auto record = dbManager_->getById(imageId);

    QString info = QString("ID: %1\nPath: %2\nSize: %3x%4")
        .arg(record.id)
        .arg(QString::fromStdString(record.filePath))
        .arg(record.width)
        .arg(record.height);

    statusLabel_->setText(info);
}

void ImageSearchWidget::onResultDoubleClicked(int64_t imageId) {
    auto record = dbManager_->getById(imageId);

    if (record.id < 0) {
        showError("Image record not found");
        return;
    }

    // 打开图像文件
    QString filePath = QString::fromStdString(record.filePath);
    QDesktopServices::openUrl(QUrl::fromLocalFile(filePath));
}

void ImageSearchWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
    emit errorOccurred(message);
    statusLabel_->setText("Error: " + message);
}

} // namespace gui
} // namespace vindex
