#include "api_ai_widget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QImageReader>
#include <QBuffer>
#include <QByteArray>
#include <QJsonArray>
#include <QJsonObject>
#include <QJsonDocument>
#include <QMessageBox>
#include <QLabel>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QImage>
#include <QPixmap>
#include <QSize>
#include <QIODevice>
#include <QStringList>

namespace vindex {
namespace gui {

ApiAIWidget::ApiAIWidget(core::ModelManager* modelManager, QWidget* parent)
    : QWidget(parent)
    , modelManager_(modelManager) {
    setupUI();
}

void ApiAIWidget::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);

    // 文生图
    auto* imgGroup = new QGroupBox("Text to Image (BigModel images/generations)", this);
    auto* imgLayout = new QVBoxLayout(imgGroup);

    promptEdit_ = new QPlainTextEdit(this);
    promptEdit_->setPlaceholderText("请输入生成描述，例如：一只可爱的小猫咪，坐在阳光明媚的窗台上...");
    promptEdit_->setMaximumHeight(80);
    imgLayout->addWidget(promptEdit_);

    auto* optLayout = new QHBoxLayout();
    modelEditImg_ = new QLineEdit("cogView-4-250304", this);
    modelPresetImg_ = new QComboBox(this);
    modelPresetImg_->addItems({"cogView-4-250304", "wanx-v1", "wanx-v1.1"});
    connect(modelPresetImg_, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        modelEditImg_->setText(text);
    });
    sizeCombo_ = new QComboBox(this);
    sizeCombo_->addItems({"1024x1024", "768x768", "512x512"});
    qualityCombo_ = new QComboBox(this);
    qualityCombo_->addItems({"standard", "hd"});
    optLayout->addWidget(new QLabel("Model:", this));
    optLayout->addWidget(modelPresetImg_, 1);
    optLayout->addWidget(modelEditImg_, 1);
    optLayout->addWidget(new QLabel("Size:", this));
    optLayout->addWidget(sizeCombo_);
    optLayout->addWidget(new QLabel("Quality:", this));
    optLayout->addWidget(qualityCombo_);
    imgLayout->addLayout(optLayout);

    auto* tokenLayout = new QHBoxLayout();
    tokenEdit_ = new QLineEdit(this);
    tokenEdit_->setPlaceholderText("Bearer token (qwen/glm)");
    tokenEdit_->setEchoMode(QLineEdit::Password);
    tokenLayout->addWidget(new QLabel("Token:", this));
    tokenLayout->addWidget(tokenEdit_);
    genBtn_ = new QPushButton("Generate", this);
    connect(genBtn_, &QPushButton::clicked, this, &ApiAIWidget::onGenerateImage);
    tokenLayout->addWidget(genBtn_);
    imgLayout->addLayout(tokenLayout);

    imageLabel_ = new QLabel(this);
    imageLabel_->setFixedSize(320, 320);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet("QLabel { background:#f5f5f5; border:1px solid #ccc; }");
    imageLabel_->setText("Generated image will appear here");
    imgLayout->addWidget(imageLabel_);

    mainLayout->addWidget(imgGroup);

    // 图生文 / VQA
    auto* vqaGroup = new QGroupBox("Image to Text (chat/completions with image)", this);
    auto* vqaLayout = new QVBoxLayout(vqaGroup);

    auto* topLayout = new QHBoxLayout();
    vqaImageLabel_ = new QLabel(this);
    vqaImageLabel_->setFixedSize(200, 200);
    vqaImageLabel_->setAlignment(Qt::AlignCenter);
    vqaImageLabel_->setStyleSheet("QLabel { background:#f5f5f5; border:1px dashed #ccc; }");
    vqaImageLabel_->setText("No image");
    topLayout->addWidget(vqaImageLabel_);

    auto* rightLayout = new QVBoxLayout();
    selectImgBtn_ = new QPushButton("Select Image", this);
    connect(selectImgBtn_, &QPushButton::clicked, this, &ApiAIWidget::onSelectImage);
    rightLayout->addWidget(selectImgBtn_);

    questionEdit_ = new QLineEdit(this);
    questionEdit_->setPlaceholderText("请输入问题/描述请求，如：描述这张图片的场景");
    rightLayout->addWidget(questionEdit_);

    modelEditVqa_ = new QLineEdit("glm-4v", this);
    modelPresetVqa_ = new QComboBox(this);
    modelPresetVqa_->addItems({"glm-4v", "qwen-vl-plus", "qwen-vl-max"});
    connect(modelPresetVqa_, &QComboBox::currentTextChanged, this, [this](const QString& text) {
        modelEditVqa_->setText(text);
    });
    rightLayout->addWidget(new QLabel("Model:", this));
    rightLayout->addWidget(modelPresetVqa_);
    rightLayout->addWidget(modelEditVqa_);

    tokenVqaEdit_ = new QLineEdit(this);
    tokenVqaEdit_->setPlaceholderText("Bearer token");
    tokenVqaEdit_->setEchoMode(QLineEdit::Password);
    rightLayout->addWidget(new QLabel("Token:", this));
    rightLayout->addWidget(tokenVqaEdit_);

    askBtn_ = new QPushButton("Ask", this);
    connect(askBtn_, &QPushButton::clicked, this, &ApiAIWidget::onAskImage);
    rightLayout->addWidget(askBtn_);
    rightLayout->addStretch();

    topLayout->addLayout(rightLayout);
    vqaLayout->addLayout(topLayout);

    answerLabel_ = new QLabel("Answer will appear here", this);
    answerLabel_->setWordWrap(true);
    vqaLayout->addWidget(answerLabel_);

    mainLayout->addWidget(vqaGroup);
    mainLayout->addStretch();
}

void ApiAIWidget::onGenerateImage() {
    QString token = tokenEdit_->text().trimmed();
    if (token.isEmpty()) {
        showError("请先填入 Token");
        return;
    }
    QString prompt = promptEdit_->toPlainText().trimmed();
    if (prompt.isEmpty()) {
        showError("请输入生成描述");
        return;
    }
    QString url = "https://open.bigmodel.cn/api/paas/v4/images/generations";
    QJsonObject payload{
        {"model", modelEditImg_->text().trimmed()},
        {"prompt", prompt},
        {"size", sizeCombo_->currentText()},
        {"quality", qualityCombo_->currentText()}
    };
    genBtn_->setEnabled(false);
    imageLabel_->setText("Generating...");
    apiClient_.postJson(url, payload, token,
        [this](const QJsonDocument& doc) { handleImageResponse(doc); },
        [this](const QString& err) {
            genBtn_->setEnabled(true);
            showError(QString("Generate failed: %1").arg(err));
        });
}

void ApiAIWidget::handleImageResponse(const QJsonDocument& doc) {
    genBtn_->setEnabled(true);
    if (!doc.isObject()) {
        showError("Invalid response");
        return;
    }
    auto obj = doc.object();
    QImage img;
    if (obj.contains("data") && obj["data"].isArray()) {
        auto arr = obj["data"].toArray();
        if (!arr.isEmpty()) {
            auto item = arr.first().toObject();
            if (item.contains("b64_json")) {
                QByteArray b64 = item["b64_json"].toString().toUtf8();
                QByteArray raw = QByteArray::fromBase64(b64);
                img.loadFromData(raw);
            } else if (item.contains("url")) {
                // 简单处理：提示 URL
                imageLabel_->setText(QString("Image URL: %1").arg(item["url"].toString()));
                return;
            }
        }
    }
    if (img.isNull()) {
        showError("No image in response");
        return;
    }
    imageLabel_->setPixmap(QPixmap::fromImage(img).scaled(imageLabel_->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ApiAIWidget::onSelectImage() {
    QString file = QFileDialog::getOpenFileName(
        this,
        "Select Image",
        QString(),
        "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)");
    if (file.isEmpty()) return;
    currentImagePath_ = file;
    QImageReader reader(file);
    reader.setScaledSize(QSize(200, 200));
    QImage img = reader.read();
    if (img.isNull()) {
        showError("Failed to load image");
        return;
    }
    vqaImageLabel_->setPixmap(QPixmap::fromImage(img));
}

QByteArray ApiAIWidget::loadImageBase64(const QString& path) {
    QImage img(path);
    if (img.isNull()) return {};
    QByteArray ba;
    QBuffer buf(&ba);
    buf.open(QIODevice::WriteOnly);
    img.save(&buf, "PNG");
    return ba.toBase64();
}

void ApiAIWidget::onAskImage() {
    if (currentImagePath_.isEmpty()) {
        showError("请先选择图片");
        return;
    }
    QString token = tokenVqaEdit_->text().trimmed();
    if (token.isEmpty()) {
        showError("请先填入 Token");
        return;
    }
    QString question = questionEdit_->text().trimmed();
    if (question.isEmpty()) {
        question = "描述这张图片";
    }

    QByteArray b64 = loadImageBase64(currentImagePath_);
    if (b64.isEmpty()) {
        showError("无法读取图片");
        return;
    }

    QString url = "https://open.bigmodel.cn/api/paas/v4/chat/completions";
    QJsonObject imageObj{
        {"type", "image_url"},
        {"image_url", QJsonObject{{"url", QString("data:image/png;base64,%1").arg(QString::fromUtf8(b64))}}}
    };
    QJsonObject textObj{
        {"type", "text"},
        {"text", question}
    };
    QJsonArray content;
    content.append(textObj);
    content.append(imageObj);

    QJsonObject message{
        {"role", "user"},
        {"content", content}
    };

    QJsonObject payload{
        {"model", modelEditVqa_->text().trimmed()},
        {"messages", QJsonArray{message}}
    };

    askBtn_->setEnabled(false);
    answerLabel_->setText("Asking...");

    apiClient_.postJson(url, payload, token,
        [this](const QJsonDocument& doc) { handleVqaResponse(doc); },
        [this](const QString& err) {
            askBtn_->setEnabled(true);
            showError(QString("Ask failed: %1").arg(err));
        });
}

void ApiAIWidget::handleVqaResponse(const QJsonDocument& doc) {
    askBtn_->setEnabled(true);
    if (!doc.isObject()) {
        showError("Invalid response");
        return;
    }
    auto obj = doc.object();
    if (!obj.contains("choices")) {
        showError("No choices in response");
        return;
    }
    auto arr = obj["choices"].toArray();
    if (arr.isEmpty()) {
        showError("Empty choices");
        return;
    }
    auto msgObj = arr.first().toObject()["message"].toObject();
    QString content;
    if (msgObj.contains("content")) {
        auto c = msgObj["content"];
        if (c.isArray()) {
            // 可能是多段
            QStringList parts;
            for (auto v : c.toArray()) {
                if (v.isObject() && v.toObject().value("type") == "text") {
                    parts << v.toObject().value("text").toString();
                }
            }
            content = parts.join("\n");
        } else {
            content = c.toString();
        }
    }
    if (content.isEmpty()) {
        showError("No content in response");
        return;
    }
    answerLabel_->setText(content);
}

void ApiAIWidget::showError(const QString& msg) {
    QMessageBox::warning(this, "Error", msg);
    answerLabel_->setText("Error: " + msg);
}

} // namespace gui
} // namespace vindex
