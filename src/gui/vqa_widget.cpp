#include "vqa_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>
#include <opencv2/opencv.hpp>

namespace vindex {
namespace gui {

VQAWidget::VQAWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();
}

void VQAWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(320, 320);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    imageLabel_->setText("No image selected");

    selectBtn_ = new QPushButton("Select Image", this);
    connect(selectBtn_, &QPushButton::clicked, this, &VQAWidget::onSelectImage);

    askBtn_ = new QPushButton("Ask", this);
    connect(askBtn_, &QPushButton::clicked, this, &VQAWidget::onAsk);

    questionEdit_ = new QLineEdit(this);
    questionEdit_->setPlaceholderText("Ask a question about the image");

    auto* topLayout = new QHBoxLayout();
    topLayout->addWidget(imageLabel_);
    auto* sideLayout = new QVBoxLayout();
    sideLayout->addWidget(questionEdit_);
    sideLayout->addWidget(askBtn_);
    sideLayout->addWidget(selectBtn_);
    sideLayout->addStretch();
    topLayout->addLayout(sideLayout);
    mainLayout->addLayout(topLayout);

    answerLabel_ = new QLabel("Answer: N/A", this);
    answerLabel_->setWordWrap(true);
    mainLayout->addWidget(answerLabel_);
    mainLayout->addStretch();
}

void VQAWidget::onSelectImage() {
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

void VQAWidget::onAsk() {
    if (currentImagePath_.isEmpty()) {
        showError("Please select an image first");
        return;
    }
    if (questionEdit_->text().trimmed().isEmpty()) {
        showError("Please enter a question");
        return;
    }

    try {
        auto& vqaModel = modelManager_->vqaModel();
        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        std::string answer = vqaModel.answer(image, questionEdit_->text().toStdString());
        answerLabel_->setText(QString("Answer: %1").arg(QString::fromStdString(answer)));
    } catch (const std::exception& e) {
        showError(QString("VQA failed: %1").arg(e.what()));
    }
}

void VQAWidget::showError(const QString& message) {
    QMessageBox::warning(this, "Error", message);
}

} // namespace gui
} // namespace vindex
