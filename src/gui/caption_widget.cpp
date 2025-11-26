#include "caption_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>
#include <opencv2/opencv.hpp>

namespace vindex {
namespace gui {

CaptionWidget::CaptionWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();
}

void CaptionWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(320, 320);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    imageLabel_->setText("No image selected");

    selectBtn_ = new QPushButton("Select Image", this);
    connect(selectBtn_, &QPushButton::clicked, this, &CaptionWidget::onSelectImage);

    generateBtn_ = new QPushButton("Generate Caption", this);
    connect(generateBtn_, &QPushButton::clicked, this, &CaptionWidget::onGenerate);

    auto* topLayout = new QHBoxLayout();
    topLayout->addWidget(imageLabel_);
    auto* btnLayout = new QVBoxLayout();
    btnLayout->addWidget(selectBtn_);
    btnLayout->addWidget(generateBtn_);
    btnLayout->addStretch();
    topLayout->addLayout(btnLayout);

    mainLayout->addLayout(topLayout);

    captionLabel_ = new QLabel("Caption: N/A", this);
    captionLabel_->setWordWrap(true);
    captionLabel_->setStyleSheet("QLabel { font-size: 14px; }");
    mainLayout->addWidget(captionLabel_);
    mainLayout->addStretch();
}

void CaptionWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Select Image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)"
    );

    if (fileName.isEmpty()) return;

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

void CaptionWidget::onGenerate() {
    if (currentImagePath_.isEmpty()) {
        showError("Please select an image first");
        return;
    }

    try {
        auto& captionModel = modelManager_->captionModel();
        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        std::string text = captionModel.generate(image);
        captionLabel_->setText(QString("Caption: %1").arg(QString::fromStdString(text)));
    } catch (const std::exception& e) {
        showError(QString("Caption failed: %1").arg(e.what()));
    }
}

void CaptionWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
}

} // namespace gui
} // namespace vindex
