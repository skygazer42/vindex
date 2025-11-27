#include "ocr_widget.h"
#include "../utils/translator.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>
#include <QClipboard>
#include <QApplication>
#include <opencv2/opencv.hpp>

namespace vindex {
namespace gui {

OcrWidget::OcrWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();

    // 连接语言切换信号
    connect(&utils::Translator::instance(), &utils::Translator::languageChanged,
            this, &OcrWidget::retranslateUI);
}

void OcrWidget::setupUI() {
    auto* mainLayout = new QHBoxLayout(this);

    // 左侧：输入图像
    inputGroup_ = new QGroupBox(TR("Input Image"), this);
    auto* inputLayout = new QVBoxLayout(inputGroup_);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(400, 400);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    imageLabel_->setText(TR("No image selected"));
    imageLabel_->setScaledContents(false);
    inputLayout->addWidget(imageLabel_, 0, Qt::AlignCenter);

    auto* btnLayout = new QHBoxLayout();
    selectBtn_ = new QPushButton(TR("Select Image"), this);
    connect(selectBtn_, &QPushButton::clicked, this, &OcrWidget::onSelectImage);
    btnLayout->addWidget(selectBtn_);

    recognizeBtn_ = new QPushButton(TR("Recognize"), this);
    connect(recognizeBtn_, &QPushButton::clicked, this, &OcrWidget::onRecognize);
    btnLayout->addWidget(recognizeBtn_);

    inputLayout->addLayout(btnLayout);
    mainLayout->addWidget(inputGroup_);

    // 右侧：识别结果
    outputGroup_ = new QGroupBox(TR("Recognition Result"), this);
    auto* outputLayout = new QVBoxLayout(outputGroup_);

    resultText_ = new QTextEdit(this);
    resultText_->setReadOnly(true);
    resultText_->setPlaceholderText(TR("OCR result will appear here..."));
    resultText_->setStyleSheet("QTextEdit { font-size: 14px; }");
    outputLayout->addWidget(resultText_);

    copyBtn_ = new QPushButton(TR("Copy"), this);
    connect(copyBtn_, &QPushButton::clicked, this, &OcrWidget::onCopyText);
    outputLayout->addWidget(copyBtn_);

    mainLayout->addWidget(outputGroup_);
}

void OcrWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        TR("Select Query Image"),
        QString(),
        TR("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)")
    );

    if (fileName.isEmpty()) return;

    QImageReader reader(fileName);
    QSize imageSize = reader.size();

    // 计算缩放后的尺寸以适应标签
    int maxSize = 400;
    if (imageSize.width() > maxSize || imageSize.height() > maxSize) {
        float ratio = std::min(
            static_cast<float>(maxSize) / imageSize.width(),
            static_cast<float>(maxSize) / imageSize.height()
        );
        imageSize.setWidth(static_cast<int>(imageSize.width() * ratio));
        imageSize.setHeight(static_cast<int>(imageSize.height() * ratio));
    }

    reader.setScaledSize(imageSize);
    QImage image = reader.read();
    if (image.isNull()) {
        showError(TR("Failed to load image"));
        return;
    }

    imageLabel_->setPixmap(QPixmap::fromImage(image));
    currentImagePath_ = fileName;
    resultText_->clear();
}

void OcrWidget::onRecognize() {
    if (currentImagePath_.isEmpty()) {
        showError(TR("Please select an image"));
        return;
    }

    // 检查模型是否加载
    if (!modelManager_->hasOcrModel()) {
        showError(TR("OCR model not loaded"));
        return;
    }

    try {
        recognizeBtn_->setEnabled(false);
        recognizeBtn_->setText(TR("Recognizing..."));

        auto& ocrModel = modelManager_->ocrModel();
        if (!ocrModel.loaded()) {
            showError(TR("OCR model not loaded"));
            recognizeBtn_->setEnabled(true);
            recognizeBtn_->setText(TR("Recognize"));
            return;
        }

        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        if (image.empty()) {
            showError(TR("Failed to load image"));
            recognizeBtn_->setEnabled(true);
            recognizeBtn_->setText(TR("Recognize"));
            return;
        }

        std::string text = ocrModel.recognizeText(image);
        resultText_->setText(QString::fromStdString(text));

        recognizeBtn_->setEnabled(true);
        recognizeBtn_->setText(TR("Recognize"));

    } catch (const std::exception& e) {
        recognizeBtn_->setEnabled(true);
        recognizeBtn_->setText(TR("Recognize"));
        showError(QString(TR("Search failed: %1")).arg(e.what()));
    }
}

void OcrWidget::onCopyText() {
    QString text = resultText_->toPlainText();
    if (!text.isEmpty()) {
        QClipboard* clipboard = QApplication::clipboard();
        clipboard->setText(text);
    }
}

void OcrWidget::retranslateUI() {
    inputGroup_->setTitle(TR("Input Image"));
    outputGroup_->setTitle(TR("Recognition Result"));
    selectBtn_->setText(TR("Select Image"));
    recognizeBtn_->setText(TR("Recognize"));
    copyBtn_->setText(TR("Copy"));
    resultText_->setPlaceholderText(TR("OCR result will appear here..."));

    if (currentImagePath_.isEmpty()) {
        imageLabel_->setText(TR("No image selected"));
    }
}

void OcrWidget::showError(const QString& message) {
    QMessageBox::warning(this, TR("Error"), message);
}

} // namespace gui
} // namespace vindex
