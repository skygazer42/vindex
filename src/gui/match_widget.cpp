#include "match_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QImageReader>
#include <QPixmap>
#include <opencv2/opencv.hpp>

namespace vindex {
namespace gui {

MatchWidget::MatchWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();
}

void MatchWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // 图像区域
    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(320, 320);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    imageLabel_->setText("No image selected");

    selectBtn_ = new QPushButton("Select Image", this);
    connect(selectBtn_, &QPushButton::clicked, this, &MatchWidget::onSelectImage);

    auto* imageLayout = new QVBoxLayout();
    imageLayout->addWidget(imageLabel_);
    imageLayout->addWidget(selectBtn_);

    // 文本输入与计算
    textEdit_ = new QLineEdit(this);
    textEdit_->setPlaceholderText("Enter text to match");
    computeBtn_ = new QPushButton("Compute Similarity", this);
    connect(computeBtn_, &QPushButton::clicked, this, &MatchWidget::onCompute);

    auto* textLayout = new QVBoxLayout();
    textLayout->addWidget(new QLabel("Text:", this));
    textLayout->addWidget(textEdit_);
    textLayout->addWidget(computeBtn_);

    auto* topLayout = new QHBoxLayout();
    topLayout->addLayout(imageLayout);
    topLayout->addLayout(textLayout, 1);
    mainLayout->addLayout(topLayout);

    scoreLabel_ = new QLabel("Score: N/A", this);
    scoreLabel_->setStyleSheet("QLabel { font-weight: bold; }");
    mainLayout->addWidget(scoreLabel_);
    mainLayout->addStretch();
}

void MatchWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    );

    if (fileName.isEmpty()) {
        return;
    }

    QImageReader reader(fileName);
    reader.setScaledSize(QSize(320, 320));
    QImage image = reader.read();
    if (image.isNull()) {
        showError("Failed to load image");
        return;
    }

    imageLabel_->setPixmap(QPixmap::fromImage(image));
    currentImagePath_ = fileName;
}

void MatchWidget::onCompute() {
    if (currentImagePath_.isEmpty()) {
        showError("Please select an image");
        return;
    }
    if (textEdit_->text().trimmed().isEmpty()) {
        showError("Please enter text");
        return;
    }

    try {
        auto& encoder = modelManager_->clipEncoder();
        if (!encoder.hasTextEncoder()) {
            showError("Text encoder not loaded. Please place clip_text.onnx and vocab.");
            return;
        }

        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        float score = encoder.computeSimilarity(image, textEdit_->text().toStdString());
        scoreLabel_->setText(QString("Score: %1").arg(score, 0, 'f', 3));

    } catch (const std::exception& e) {
        showError(QString("Failed to compute: %1").arg(e.what()));
    }
}

void MatchWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
    scoreLabel_->setText("Score: Error");
}

} // namespace gui
} // namespace vindex
