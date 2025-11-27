#pragma once

#include <QObject>
#include <QNetworkAccessManager>
#include <QJsonObject>
#include <QJsonDocument>
#include <QHash>
#include <QNetworkReply>
#include <functional>

namespace vindex {
namespace utils {

/**
 * @brief 轻量 HTTP JSON 客户端，用于调用外部模型 API（如 BigModel 文生图/图生文）
 */
class ApiClient : public QObject {
    Q_OBJECT
public:
    explicit ApiClient(QObject* parent = nullptr);
    ~ApiClient() = default;

    /**
     * @brief POST JSON 请求
     * @param url API 地址
     * @param payload JSON 负载
     * @param bearerToken 授权 token（会以 Bearer 形式放入 Authorization）
     * @param onSuccess 成功回调（返回解析后的 JSON）
     * @param onError 失败回调（错误字符串）
     */
    void postJson(const QString& url,
                  const QJsonObject& payload,
                  const QString& bearerToken,
                  std::function<void(const QJsonDocument&)> onSuccess,
                  std::function<void(const QString&)> onError);

private slots:
    void handleReply(QNetworkReply* reply);

private:
    QNetworkAccessManager manager_;
    // 简单映射 reply 指针到回调
    struct Callbacks {
        std::function<void(const QJsonDocument&)> onSuccess;
        std::function<void(const QString&)> onError;
    };
    QHash<QNetworkReply*, Callbacks> pending_;
};

} // namespace utils
} // namespace vindex
