#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QPlainTextEdit>
#include <QLineEdit>
#include <QComboBox>
#include <QPixmap>
#include <QByteArray>
#include <QNetworkAccessManager>
#include <QJsonDocument>
#include "../utils/api_client.h"
#include "../core/model_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 远程 API 集成（文生图 / 图生文 VQA）
 *
 * 文生图：调用 BigModel images/generations 接口
 * 图生文（简易 VQA/描述）：调用 chat/completions，附带图像 base64
 */
class ApiAIWidget : public QWidget {
    Q_OBJECT
public:
    explicit ApiAIWidget(core::ModelManager* modelManager, QWidget* parent = nullptr);
    ~ApiAIWidget() = default;

private slots:
    void onGenerateImage();
    void onSelectImage();
    void onAskImage();

private:
    void setupUI();
    void handleImageResponse(const QJsonDocument& doc);
    void handleVqaResponse(const QJsonDocument& doc);
    QByteArray loadImageBase64(const QString& path);
    void showError(const QString& msg);

private:
    core::ModelManager* modelManager_;
    utils::ApiClient apiClient_;

    // 文生图
    QPlainTextEdit* promptEdit_;
    QLineEdit* modelEditImg_;
    QComboBox* sizeCombo_;
    QComboBox* qualityCombo_;
    QLineEdit* tokenEdit_;
    QPushButton* genBtn_;
    QLabel* imageLabel_;
    QComboBox* modelPresetImg_;

    // 图生文
    QLabel* vqaImageLabel_;
    QPushButton* selectImgBtn_;
    QLineEdit* questionEdit_;
    QComboBox* modelPresetVqa_;
    QLineEdit* modelEditVqa_;
    QLineEdit* tokenVqaEdit_;
    QPushButton* askBtn_;
    QLabel* answerLabel_;
    QString currentImagePath_;
};

} // namespace gui
} // namespace vindex
