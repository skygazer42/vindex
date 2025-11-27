#include "vqa_widget.h"
#include "../utils/translator.h"
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

    // 连接语言切换信号
    connect(&utils::Translator::instance(), &utils::Translator::languageChanged,
            this, &VQAWidget::retranslateUI);
}

void VQAWidget::setupUI() {
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
    imageLabel_->setText(TR("No image selected"));
    inputLayout->addWidget(imageLabel_, 0, Qt::AlignCenter);

    selectBtn_ = new QPushButton(TR("Select Image"), this);
    connect(selectBtn_, &QPushButton::clicked, this, &VQAWidget::onSelectImage);
    inputLayout->addWidget(selectBtn_);

    mainLayout->addWidget(inputGroup_);

    // 问题输入区域
    questionGroup_ = new QGroupBox(TR("Question:"), this);
    auto* questionLayout = new QHBoxLayout(questionGroup_);

    questionEdit_ = new QLineEdit(this);
    questionEdit_->setPlaceholderText(TR("Ask a question about the image"));
    questionLayout->addWidget(questionEdit_);

    askBtn_ = new QPushButton(TR("Ask"), this);
    connect(askBtn_, &QPushButton::clicked, this, &VQAWidget::onAsk);
    questionLayout->addWidget(askBtn_);

    mainLayout->addWidget(questionGroup_);

    // 答案输出区域
    outputGroup_ = new QGroupBox(TR("Answer"), this);
    auto* outputLayout = new QVBoxLayout(outputGroup_);

    answerLabel_ = new QLabel(TR("Answer will appear here..."), this);
    answerLabel_->setWordWrap(true);
    answerLabel_->setStyleSheet("QLabel { font-size: 14px; padding: 10px; background-color: #fafafa; border-radius: 4px; }");
    answerLabel_->setMinimumHeight(80);
    outputLayout->addWidget(answerLabel_);

    mainLayout->addWidget(outputGroup_);
    mainLayout->addStretch();
}

void VQAWidget::onSelectImage() {
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
    answerLabel_->setText(TR("Answer will appear here..."));
}

void VQAWidget::onAsk() {
    if (currentImagePath_.isEmpty()) {
        showError(TR("Please select an image"));
        return;
    }
    if (questionEdit_->text().trimmed().isEmpty()) {
        showError(TR("Please enter a question"));
        return;
    }

    // 检查模型是否加载
    if (!modelManager_->hasVqaModel()) {
        showError(TR("VQA model not loaded"));
        return;
    }

    try {
        askBtn_->setEnabled(false);
        askBtn_->setText("...");

        auto& vqaModel = modelManager_->vqaModel();
        if (!vqaModel.loaded()) {
            showError(TR("VQA model not loaded"));
            askBtn_->setEnabled(true);
            askBtn_->setText(TR("Ask"));
            return;
        }

        cv::Mat image = cv::imread(currentImagePath_.toStdString());
        if (image.empty()) {
            showError(TR("Failed to load image"));
            askBtn_->setEnabled(true);
            askBtn_->setText(TR("Ask"));
            return;
        }

        std::string answer = vqaModel.answer(image, questionEdit_->text().toStdString());
        answerLabel_->setText(QString::fromStdString(answer));

        askBtn_->setEnabled(true);
        askBtn_->setText(TR("Ask"));

    } catch (const std::exception& e) {
        askBtn_->setEnabled(true);
        askBtn_->setText(TR("Ask"));
        showError(QString(TR("Search failed: %1")).arg(e.what()));
    }
}

void VQAWidget::retranslateUI() {
    inputGroup_->setTitle(TR("Input Image"));
    questionGroup_->setTitle(TR("Question:"));
    outputGroup_->setTitle(TR("Answer"));
    selectBtn_->setText(TR("Select Image"));
    askBtn_->setText(TR("Ask"));
    questionEdit_->setPlaceholderText(TR("Ask a question about the image"));

    if (currentImagePath_.isEmpty()) {
        imageLabel_->setText(TR("No image selected"));
    }
}

void VQAWidget::showError(const QString& message) {
    QMessageBox::warning(this, TR("Error"), message);
}

} // namespace gui
} // namespace vindex
