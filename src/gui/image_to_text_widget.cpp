#include "image_to_text_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>
#include <QPixmap>
#include <QAbstractItemView>
#include <QCoreApplication>
#include <QFileInfo>
#include <QStringList>
#include <QTextEdit>

namespace vindex {
namespace gui {

ImageToTextWidget::ImageToTextWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager)
    , corpus_(modelManager ? modelManager->getEmbeddingDim() : 512)
    , corpusReady_(false) {
    setupUI();
    loadCorpus();
}

void ImageToTextWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    auto* topGroup = new QGroupBox("Query Image", this);
    auto* topLayout = new QHBoxLayout(topGroup);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(300, 300);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet("QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }");
    imageLabel_->setText("No image selected");

    auto* controlLayout = new QVBoxLayout();

    selectBtn_ = new QPushButton("Select Image", this);
    connect(selectBtn_, &QPushButton::clicked, this, &ImageToTextWidget::onSelectImage);
    controlLayout->addWidget(selectBtn_);

    auto* topKLayout = new QHBoxLayout();
    topKLayout->addWidget(new QLabel("Top K:", this));
    topKSpinBox_ = new QSpinBox(this);
    topKSpinBox_->setRange(1, 20);
    topKSpinBox_->setValue(5);
    topKLayout->addWidget(topKSpinBox_);
    topKLayout->addStretch();
    controlLayout->addLayout(topKLayout);

    auto* thLayout = new QHBoxLayout();
    thLayout->addWidget(new QLabel("Threshold:", this));
    thresholdEdit_ = new QLineEdit("0.3", this);
    thresholdEdit_->setMaximumWidth(80);
    thLayout->addWidget(thresholdEdit_);
    thLayout->addStretch();
    controlLayout->addLayout(thLayout);

    searchBtn_ = new QPushButton("Search", this);
    searchBtn_->setEnabled(false);
    connect(searchBtn_, &QPushButton::clicked, this, &ImageToTextWidget::onSearch);
    controlLayout->addWidget(searchBtn_);
    controlLayout->addStretch();

    topLayout->addWidget(imageLabel_);
    topLayout->addLayout(controlLayout);
    mainLayout->addWidget(topGroup);

    statusLabel_ = new QLabel("Ready", this);
    mainLayout->addWidget(statusLabel_);

    // 文本查询（文搜文）
    auto* textGroup = new QGroupBox("Text Query", this);
    auto* textLayout = new QVBoxLayout(textGroup);
    textQueryEdit_ = new QTextEdit(this);
    textQueryEdit_->setPlaceholderText("Enter text to search in corpus");
    textQueryEdit_->setMaximumHeight(80);
    textLayout->addWidget(textQueryEdit_);
    searchTextBtn_ = new QPushButton("Search Text", this);
    searchTextBtn_->setEnabled(false);
    connect(searchTextBtn_, &QPushButton::clicked, this, &ImageToTextWidget::onSearchText);
    textLayout->addWidget(searchTextBtn_);
    mainLayout->addWidget(textGroup);

    auto* resultGroup = new QGroupBox("Matched Captions (examples)", this);
    auto* resultLayout = new QVBoxLayout(resultGroup);
    resultList_ = new QListWidget(this);
    resultList_->setSelectionMode(QAbstractItemView::NoSelection);
    resultLayout->addWidget(resultList_);
    mainLayout->addWidget(resultGroup, 1);
}

void ImageToTextWidget::loadCorpus() {
    if (!modelManager_) {
        showError("Model manager not initialized");
        return;
    }
    auto findCorpus = []() -> QString {
        QStringList candidates = {
            "resources/text_corpus.txt",
            QCoreApplication::applicationDirPath() + "/resources/text_corpus.txt",
            QCoreApplication::applicationDirPath() + "/../resources/text_corpus.txt"
        };
        for (const auto& c : candidates) {
            if (QFileInfo::exists(c)) return c;
        }
        return {};
    };

    try {
        auto& encoder = modelManager_->clipEncoder();
        if (!encoder.hasTextEncoder()) {
            showError("Text encoder not loaded, cannot build text corpus index.");
            selectBtn_->setEnabled(false);
            searchBtn_->setEnabled(false);
            searchTextBtn_->setEnabled(false);
            return;
        }
        QString path = findCorpus();
        if (path.isEmpty()) {
            showError("Text corpus file not found (expected resources/text_corpus.txt).");
            searchTextBtn_->setEnabled(false);
            return;
        }
        corpusReady_ = corpus_.loadFromFile(path.toStdString(), encoder);
        if (!corpusReady_) {
            showError("Failed to build text corpus index.");
            searchTextBtn_->setEnabled(false);
        } else {
            statusLabel_->setText(QString("Corpus loaded: %1 entries").arg(static_cast<int>(corpus_.size())));
            searchTextBtn_->setEnabled(true);
        }
    } catch (const std::exception& e) {
        showError(QString("Failed to load corpus: %1").arg(e.what()));
        searchTextBtn_->setEnabled(false);
    }
}

void ImageToTextWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    );
    if (fileName.isEmpty()) return;
    currentImagePath_ = fileName;
    displayImage(fileName);
    searchBtn_->setEnabled(corpusReady_);
    statusLabel_->setText("Image loaded, ready to search.");
}

void ImageToTextWidget::displayImage(const QString& path) {
    QImageReader reader(path);
    reader.setScaledSize(QSize(300, 300));
    QImage img = reader.read();
    if (img.isNull()) {
        showError("Failed to load image.");
        return;
    }
    imageLabel_->setPixmap(QPixmap::fromImage(img));
}

void ImageToTextWidget::onSearch() {
    if (!corpusReady_) {
        showError("Corpus not ready.");
        return;
    }
    if (currentImagePath_.isEmpty()) {
        showError("Please select an image first.");
        return;
    }
    try {
        auto& encoder = modelManager_->clipEncoder();
        cv::Mat img = cv::imread(currentImagePath_.toStdString());
        auto imageFeat = encoder.encodeImage(img);
        int topK = topKSpinBox_->value();
        float threshold = thresholdEdit_->text().toFloat();
        auto results = corpus_.search(imageFeat, topK, threshold);

        resultList_->clear();
        for (const auto& r : results) {
            QString text = QString::fromStdString(r.first.text);
            QString itemStr = QString("%1  |  score: %2").arg(text).arg(r.second, 0, 'f', 3);
            resultList_->addItem(itemStr);
        }
        statusLabel_->setText(QString("Found %1 matches").arg(results.size()));
    } catch (const std::exception& e) {
        showError(QString("Search failed: %1").arg(e.what()));
    }
}

void ImageToTextWidget::onSearchText() {
    if (!corpusReady_) {
        showError("Corpus not ready.");
        return;
    }
    QString query = textQueryEdit_->toPlainText().trimmed();
    if (query.isEmpty()) {
        showError("Please enter a text query.");
        return;
    }
    try {
        auto& encoder = modelManager_->clipEncoder();
        auto textFeat = encoder.encodeText(query.toStdString());
        int topK = topKSpinBox_->value();
        float threshold = thresholdEdit_->text().toFloat();
        auto results = corpus_.search(textFeat, topK, threshold);

        resultList_->clear();
        for (const auto& r : results) {
            QString text = QString::fromStdString(r.first.text);
            QString itemStr = QString("%1  |  score: %2").arg(text).arg(r.second, 0, 'f', 3);
            resultList_->addItem(itemStr);
        }
        statusLabel_->setText(QString("Found %1 matches for text query").arg(results.size()));
    } catch (const std::exception& e) {
        showError(QString("Text search failed: %1").arg(e.what()));
    }
}

void ImageToTextWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
    statusLabel_->setText("Error: " + message);
}

} // namespace gui
} // namespace vindex
