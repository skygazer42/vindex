#include "text_search_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QMessageBox>
#include <QDesktopServices>
#include <QUrl>
#include <QFileInfo>
#include <QSettings>
#include <QSplitter>

namespace vindex {
namespace gui {

TextSearchWidget::TextSearchWidget(index::DatabaseManager* dbManager,
                                   QWidget* parent)
    : QWidget(parent)
    , dbManager_(dbManager)
{
    setupUI();
    createExamples();
    loadHistory();
}

void TextSearchWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // 创建水平分割器
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    // ==================== 左侧面板 ====================
    auto* leftPanel = new QWidget(this);
    auto* leftLayout = new QVBoxLayout(leftPanel);
    leftLayout->setContentsMargins(0, 0, 0, 0);

    // 查询输入区
    auto* queryGroup = new QGroupBox("Text Query", this);
    auto* queryLayout = new QVBoxLayout(queryGroup);

    queryTextEdit_ = new QTextEdit(this);
    queryTextEdit_->setPlaceholderText(
        "Describe what you're looking for...\n\n"
        "Examples:\n"
        "  • a cat sitting on a table\n"
        "  • sunset over the ocean\n"
        "  • red sports car"
    );
    queryTextEdit_->setMaximumHeight(100);
    connect(queryTextEdit_, &QTextEdit::textChanged,
            this, &TextSearchWidget::onQueryTextChanged);
    queryLayout->addWidget(queryTextEdit_);

    // 按钮行
    auto* buttonLayout = new QHBoxLayout();
    searchBtn_ = new QPushButton("Search", this);
    searchBtn_->setMinimumHeight(36);
    searchBtn_->setEnabled(false);
    searchBtn_->setStyleSheet(
        "QPushButton { background-color: #28a745; color: white; "
        "font-size: 14px; font-weight: bold; border-radius: 5px; }"
        "QPushButton:hover { background-color: #218838; }"
        "QPushButton:disabled { background-color: #ccc; }"
    );
    connect(searchBtn_, &QPushButton::clicked, this, &TextSearchWidget::onSearch);
    buttonLayout->addWidget(searchBtn_, 1);

    clearBtn_ = new QPushButton("Clear", this);
    clearBtn_->setMinimumHeight(36);
    connect(clearBtn_, &QPushButton::clicked, this, &TextSearchWidget::onClear);
    buttonLayout->addWidget(clearBtn_);

    queryLayout->addLayout(buttonLayout);
    leftLayout->addWidget(queryGroup);

    // 参数设置
    auto* paramsGroup = new QGroupBox("Parameters", this);
    auto* paramsLayout = new QHBoxLayout(paramsGroup);

    paramsLayout->addWidget(new QLabel("Top K:", this));
    topKSpinBox_ = new QSpinBox(this);
    topKSpinBox_->setRange(1, 100);
    topKSpinBox_->setValue(10);
    topKSpinBox_->setMinimumWidth(70);
    paramsLayout->addWidget(topKSpinBox_);

    paramsLayout->addSpacing(10);
    paramsLayout->addWidget(new QLabel("Threshold:", this));
    thresholdEdit_ = new QLineEdit("0.3", this);
    thresholdEdit_->setPlaceholderText("0.0 - 1.0");
    thresholdEdit_->setMaximumWidth(70);
    paramsLayout->addWidget(thresholdEdit_);
    paramsLayout->addStretch();

    leftLayout->addWidget(paramsGroup);

    // 示例查询
    examplesWidget_ = new QWidget(this);
    auto* examplesLayout = new QVBoxLayout(examplesWidget_);
    examplesLayout->setContentsMargins(0, 5, 0, 5);
    examplesLayout->setSpacing(5);

    auto* examplesLabel = new QLabel("<b>Quick Examples:</b>", this);
    examplesLayout->addWidget(examplesLabel);

    leftLayout->addWidget(examplesWidget_);

    // 搜索历史
    auto* historyGroup = new QGroupBox("Search History", this);
    auto* historyLayout = new QVBoxLayout(historyGroup);

    historyList_ = new QListWidget(this);
    historyList_->setMaximumHeight(150);
    connect(historyList_, &QListWidget::itemClicked,
            this, &TextSearchWidget::onHistoryClicked);
    historyLayout->addWidget(historyList_);

    clearHistoryBtn_ = new QPushButton("Clear History", this);
    connect(clearHistoryBtn_, &QPushButton::clicked, [this]() {
        searchHistory_.clear();
        historyList_->clear();
        saveHistory();
    });
    historyLayout->addWidget(clearHistoryBtn_);

    leftLayout->addWidget(historyGroup);
    leftLayout->addStretch();

    leftPanel->setMaximumWidth(350);
    splitter->addWidget(leftPanel);

    // ==================== 右侧面板（结果） ====================
    auto* rightPanel = new QWidget(this);
    auto* rightLayout = new QVBoxLayout(rightPanel);
    rightLayout->setContentsMargins(0, 0, 0, 0);

    // 状态栏
    auto* statusLayout = new QHBoxLayout();
    statusLabel_ = new QLabel("Ready - Enter a query to search", this);
    statusLabel_->setStyleSheet("QLabel { color: #666; padding: 5px; }");
    statusLayout->addWidget(statusLabel_);

    progressBar_ = new QProgressBar(this);
    progressBar_->setVisible(false);
    progressBar_->setMaximumWidth(200);
    statusLayout->addWidget(progressBar_);
    statusLayout->addStretch();

    rightLayout->addLayout(statusLayout);

    // 结果展示
    auto* resultGroup = new QGroupBox("Search Results", this);
    auto* resultLayout = new QVBoxLayout(resultGroup);

    resultGallery_ = new ImageGallery(this);
    connect(resultGallery_, &ImageGallery::itemClicked,
            this, &TextSearchWidget::onResultClicked);
    connect(resultGallery_, &ImageGallery::itemDoubleClicked,
            this, &TextSearchWidget::onResultDoubleClicked);

    resultLayout->addWidget(resultGallery_);
    rightLayout->addWidget(resultGroup, 1);

    splitter->addWidget(rightPanel);
    splitter->setStretchFactor(1, 1);

    mainLayout->addWidget(splitter);
}

void TextSearchWidget::createExamples() {
    QStringList examples = {
        "a cat",
        "dog in the park",
        "sunset over ocean",
        "red sports car",
        "person with glasses"
    };

    auto* layout = qobject_cast<QVBoxLayout*>(examplesWidget_->layout());
    if (!layout) return;

    for (const auto& example : examples) {
        auto* btn = new QPushButton(example, this);
        btn->setStyleSheet(
            "QPushButton { text-align: left; padding: 5px; "
            "border: 1px solid #ddd; background-color: #f9f9f9; }"
            "QPushButton:hover { background-color: #e9e9e9; }"
        );
        btn->setCursor(Qt::PointingHandCursor);
        connect(btn, &QPushButton::clicked, this, &TextSearchWidget::onExampleClicked);
        layout->addWidget(btn);
    }
}

void TextSearchWidget::onSearch() {
    QString queryText = queryTextEdit_->toPlainText().trimmed();
    if (queryText.isEmpty()) {
        showError("Please enter a query text");
        return;
    }

    if (!dbManager_) {
        showError("Database manager not initialized");
        return;
    }

    performSearch(queryText);
}

void TextSearchWidget::onClear() {
    queryTextEdit_->clear();
    resultGallery_->clear();
    statusLabel_->setText("Ready - Enter a query to search");
}

void TextSearchWidget::onHistoryClicked(QListWidgetItem* item) {
    if (!item) return;

    // 提取查询文本（格式："query text (N results)"）
    QString text = item->text();
    int idx = text.lastIndexOf(" (");
    if (idx > 0) {
        text = text.left(idx);
    }

    queryTextEdit_->setPlainText(text);
    onSearch();
}

void TextSearchWidget::onExampleClicked() {
    auto* btn = qobject_cast<QPushButton*>(sender());
    if (!btn) return;

    queryTextEdit_->setPlainText(btn->text());
    searchBtn_->setEnabled(true);
}

void TextSearchWidget::onQueryTextChanged() {
    bool hasText = !queryTextEdit_->toPlainText().trimmed().isEmpty();
    searchBtn_->setEnabled(hasText);
}

void TextSearchWidget::performSearch(const QString& queryText) {
    searchBtn_->setEnabled(false);
    clearBtn_->setEnabled(false);
    statusLabel_->setText("Searching...");
    progressBar_->setRange(0, 0);
    progressBar_->setVisible(true);

    try {
        int topK = topKSpinBox_->value();
        float threshold = thresholdEdit_->text().toFloat();

        auto results = dbManager_->searchByText(
            queryText.toStdString(),
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

        statusLabel_->setText(QString("Found %1 results for \"%2\"")
            .arg(results.size())
            .arg(queryText.left(30) + (queryText.length() > 30 ? "..." : "")));

        // 添加到历史
        addToHistory(queryText, results.size());

        emit searchCompleted(results.size());

    } catch (const std::exception& e) {
        showError(QString("Search failed: %1").arg(e.what()));
    }

    searchBtn_->setEnabled(true);
    clearBtn_->setEnabled(true);
    progressBar_->setVisible(false);
}

void TextSearchWidget::onResultClicked(int64_t imageId) {
    auto record = dbManager_->getById(imageId);
    QString info = QString("Selected: %1 | %2x%3 | ID: %4")
        .arg(QString::fromStdString(record.fileName))
        .arg(record.width)
        .arg(record.height)
        .arg(record.id);
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

void TextSearchWidget::addToHistory(const QString& query, int resultCount) {
    QString historyItem = QString("%1 (%2 results)").arg(query).arg(resultCount);

    // 避免重复
    for (int i = 0; i < searchHistory_.size(); ++i) {
        if (searchHistory_[i].startsWith(query + " (")) {
            searchHistory_.removeAt(i);
            break;
        }
    }

    // 添加到开头
    searchHistory_.prepend(historyItem);

    // 限制数量
    while (searchHistory_.size() > MAX_HISTORY) {
        searchHistory_.removeLast();
    }

    // 更新UI
    historyList_->clear();
    historyList_->addItems(searchHistory_);

    // 保存到设置
    saveHistory();
}

void TextSearchWidget::loadHistory() {
    QSettings settings("VIndex", "ImageSearch");
    searchHistory_ = settings.value("textSearchHistory").toStringList();

    historyList_->clear();
    historyList_->addItems(searchHistory_);
}

void TextSearchWidget::saveHistory() {
    QSettings settings("VIndex", "ImageSearch");
    settings.setValue("textSearchHistory", searchHistory_);
}

} // namespace gui
} // namespace vindex
