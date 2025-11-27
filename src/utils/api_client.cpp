#include "api_client.h"
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QJsonDocument>
#include <QByteArray>

namespace vindex {
namespace utils {

ApiClient::ApiClient(QObject* parent)
    : QObject(parent) {
    connect(&manager_, &QNetworkAccessManager::finished,
            this, &ApiClient::handleReply);
}

void ApiClient::postJson(const QString& url,
                         const QJsonObject& payload,
                         const QString& bearerToken,
                         std::function<void(const QJsonDocument&)> onSuccess,
                         std::function<void(const QString&)> onError) {
    QNetworkRequest req{QUrl(url)};
    req.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    if (!bearerToken.isEmpty()) {
        QByteArray auth = "Bearer " + bearerToken.toUtf8();
        req.setRawHeader("Authorization", auth);
    }

    QJsonDocument doc(payload);
    QByteArray body = doc.toJson(QJsonDocument::Compact);
    QNetworkReply* reply = manager_.post(req, body);

    Callbacks cb{std::move(onSuccess), std::move(onError)};
    pending_.insert(reply, std::move(cb));
}

void ApiClient::handleReply(QNetworkReply* reply) {
    auto it = pending_.find(reply);
    if (it == pending_.end()) {
        reply->deleteLater();
        return;
    }
    Callbacks cb = it.value();
    pending_.erase(it);

    if (reply->error() != QNetworkReply::NoError) {
        QString msg = reply->errorString();
        reply->deleteLater();
        if (cb.onError) cb.onError(msg);
        return;
    }

    QByteArray data = reply->readAll();
    reply->deleteLater();

    QJsonParseError parseErr;
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseErr);
    if (parseErr.error != QJsonParseError::NoError) {
        if (cb.onError) cb.onError(QString("JSON parse error: %1").arg(parseErr.errorString()));
        return;
    }
    if (cb.onSuccess) cb.onSuccess(doc);
}

} // namespace utils
} // namespace vindex

