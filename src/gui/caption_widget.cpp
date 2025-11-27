#include "caption_widget.h"
#include "../utils/translator.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QImageReader>
#include <QMessageBox>
#include <QGroupBox>
#include <opencv2/opencv.hpp>

namespace vindex {
namespace gui {

CaptionWidget::CaptionWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();

    // 连接语言切换信号
    connect(&utils::Translator::instance(), &utils::Translator::languageChanged,
            this, &CaptionWidget::retranslateUI);
}

void CaptionWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // 输入图像区域
    inputGroup_ = new QGroupBox(TR("Input Image"), this);
    auto* inputLayout = new QVBoxLayout(inputGroup_);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(320, 320);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet(
        "QLabel { background-color: #f5f5f5; border: 2px dashed #ccc; }"
    );
    imageLabel_->setText(TR("Select an image to generate caption"));
    inputLayout->addWidget(imageLabel_, 0, Qt::AlignCenter);

    auto* btnLayout = new QHBoxLayout();
    selectBtn_ = new QPushButton(TR("Select Image"), this);
    connect(selectBtn_, &QPushButton::clicked, this, &CaptionWidget::onSelectImage);
    btnLayout->addWidget(selectBtn_);

    generateBtn_ = new QPushButton(TR("Generate Caption"), this);
    connect(generateBtn_, &QPushButton::clicked, this, &CaptionWidget::onGenerate);
    btnLayout->addWidget(generateBtn_);
    btnLayout->addStretch();

    inputLayout->addLayout(btnLayout);
    mainLayout->addWidget(inputGroup_);

    // 生成结果区域
    outputGroup_ = new QGroupBox(TR("Generated Caption"), this);
    auto* outputLayout = new QVBoxLayout(outputGroup_);

    captionLabel_ = new QLabel(TR("Caption will appear here..."), this);
    captionLabel_->setWordWrap(true);
    captionLabel_->setStyleSheet("QLabel { font-size: 14px; padding: 10px; background-color: #fafafa; border-radius: 4px; }");
    captionLabel_->setMinimumHeight(80);
    outputLayout->addWidget(captionLabel_);

    mainLayout->addWidget(outputGroup_);
    mainLayout->addStretch();
}

void CaptionWidget::onSelectImage() {
    QString fileName = QFileDialog::getOpenFileName(
        this,
        TR("Select Query Image"),
        QString(),
        TR("Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;All Files (*)")
    );

    if (fileName.isEmpty()) return;

    QImageReader reader(fileName);
    reader.setScaledSize(QSize(320, 320));
    QImage image = reader.read();
    if (image.isNull()) {
        showError(TR("Failed to load image"));
        return;
    }

    imageLabel_->setPixmap(QPixmap::fromImage(image));
    currentImagePath_ = fileName;
    captionLabel_->setText(TR("Caption will appear here..."));
}

void CaptionWidget::onGenerate() {
    if (currentImagePath_.isEmpty()) {
        showError(TR("Please select an image"));
        return;
    }

    // 检查模型是否加载
    if (!modelManager_->hasCaptionModel()) {
        showError(TR("Caption model not loaded"));
        return;
    }

    try {
        generateBtn_->setEnabled(false);
        generateBtn_->setText(TR("Generating..."));

        auto& captionModel = modelManager_->captionModel();
        if (!captionModel.loaded()) {
            showError(TR("Caption model not loaded"));
            generateBtn_->setEnabled(true);
            generateBtn_->setText(TR("Generate Caption"));
            return;
        }

        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        if (image.empty()) {
            showError(TR("Failed to load image"));
            generateBtn_->setEnabled(true);
            generateBtn_->setText(TR("Generate Caption"));
            return;
        }

        std::string text = captionModel.generate(image);
        captionLabel_->setText(QString::fromStdString(text));

        generateBtn_->setEnabled(true);
        generateBtn_->setText(TR("Generate Caption"));

    } catch (const std::exception& e) {
        generateBtn_->setEnabled(true);
        generateBtn_->setText(TR("Generate Caption"));
        showError(QString(TR("Search failed: %1")).arg(e.what()));
    }
}

void CaptionWidget::retranslateUI() {
    inputGroup_->setTitle(TR("Input Image"));
    outputGroup_->setTitle(TR("Generated Caption"));
    selectBtn_->setText(TR("Select Image"));
    generateBtn_->setText(TR("Generate Caption"));

    if (currentImagePath_.isEmpty()) {
        imageLabel_->setText(TR("Select an image to generate caption"));
    }
}

void CaptionWidget::showError(const QString& message) {
    QMessageBox::warning(this, TR("Error"), message);
}

} // namespace gui
} // namespace vindex
