#include "text_search_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <QFileInfo>

namespace vindex {
namespace gui {

TextSearchWidget::TextSearchWidget(index::DatabaseManager* dbManager,
                                   QWidget* parent)
    : QWidget(parent)
    , dbManager_(dbManager) {
    setupUI();
}

void TextSearchWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    auto* queryGroup = new QGroupBox("Text Query", this);
    auto* queryLayout = new QHBoxLayout(queryGroup);

    queryEdit_ = new QLineEdit(this);
    queryEdit_->setPlaceholderText("Describe what you are looking for...");
    queryEdit_->setMinimumHeight(36);
    queryLayout->addWidget(queryEdit_, 1);

    searchBtn_ = new QPushButton("Search", this);
    searchBtn_->setMinimumHeight(36);
    searchBtn_->setStyleSheet(
        "QPushButton { background-color: #28a745; color: white; "
        "font-size: 14px; font-weight: bold; border-radius: 5px; }"
        "QPushButton:hover { background-color: #218838; }"
        "QPushButton:disabled { background-color: #ccc; }"
    );
    connect(searchBtn_, &QPushButton::clicked, this, &TextSearchWidget::onSearch);
    queryLayout->addWidget(searchBtn_);

    mainLayout->addWidget(queryGroup);

    // 参数
    auto* paramsLayout = new QHBoxLayout();
    paramsLayout->addWidget(new QLabel("Top K:", this));
    topKSpinBox_ = new QSpinBox(this);
    topKSpinBox_->setRange(1, 100);
    topKSpinBox_->setValue(10);
    paramsLayout->addWidget(topKSpinBox_);

    paramsLayout->addSpacing(12);
    paramsLayout->addWidget(new QLabel("Threshold:", this));
    thresholdEdit_ = new QLineEdit("0.0", this);
    thresholdEdit_->setMaximumWidth(80);
    paramsLayout->addWidget(thresholdEdit_);
    paramsLayout->addStretch();

    mainLayout->addLayout(paramsLayout);

    // 状态
    auto* statusLayout = new QHBoxLayout();
    statusLabel_ = new QLabel("Ready", this);
    statusLayout->addWidget(statusLabel_);
    progressBar_ = new QProgressBar(this);
    progressBar_->setVisible(false);
    progressBar_->setMaximumWidth(200);
    statusLayout->addWidget(progressBar_);
    statusLayout->addStretch();
    mainLayout->addLayout(statusLayout);

    // 结果
    auto* resultGroup = new QGroupBox("Search Results", this);
    auto* resultLayout = new QVBoxLayout(resultGroup);
    resultGallery_ = new ImageGallery(this);
    connect(resultGallery_, &ImageGallery::itemClicked,
            this, &TextSearchWidget::onResultClicked);
    connect(resultGallery_, &ImageGallery::itemDoubleClicked,
            this, &TextSearchWidget::onResultDoubleClicked);
    resultLayout->addWidget(resultGallery_);
    mainLayout->addWidget(resultGroup, 1);
}

void TextSearchWidget::onSearch() {
    if (queryEdit_->text().trimmed().isEmpty()) {
        showError("Please enter a query text");
        return;
    }

    if (!dbManager_) {
        showError("Database manager not initialized");
        return;
    }

    performSearch();
}

void TextSearchWidget::performSearch() {
    searchBtn_->setEnabled(false);
    statusLabel_->setText("Searching...");
    progressBar_->setRange(0, 0);
    progressBar_->setVisible(true);

    try {
        int topK = topKSpinBox_->value();
        float threshold = thresholdEdit_->text().toFloat();
        auto results = dbManager_->searchByText(
            queryEdit_->text().toStdString(),
            topK,
            threshold
        );

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

        resultGallery_->setResults(items);
        statusLabel_->setText(QString("Found %1 results").arg(results.size()));
        emit searchCompleted(items.size());

    } catch (const std::exception& e) {
        showError(QString("Search failed: %1").arg(e.what()));
    }

    searchBtn_->setEnabled(true);
    progressBar_->setVisible(false);
}

void TextSearchWidget::onResultClicked(int64_t imageId) {
    auto record = dbManager_->getById(imageId);
    QString info = QString("ID: %1 | %2x%3")
        .arg(record.id)
        .arg(record.width)
        .arg(record.height);
    statusLabel_->setText(info);
}

void TextSearchWidget::onResultDoubleClicked(int64_t imageId) {
    auto record = dbManager_->getById(imageId);
    if (record.id < 0) {
        showError("Image record not found");
        return;
    }

    QDesktopServices::openUrl(QUrl::fromLocalFile(
        QString::fromStdString(record.filePath)
    ));
}

void TextSearchWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
    statusLabel_->setText("Error: " + message);
    emit errorOccurred(message);
}

} // namespace gui
} // namespace vindex
