#include "database_widget.h"
#include <QVBoxLayout>
#include <QString>

namespace vindex {
namespace gui {

DatabaseWidget::DatabaseWidget(index::DatabaseManager* dbManager, QWidget* parent)
    : QWidget(parent)
    , dbManager_(dbManager) {
    setupUI();
}

void DatabaseWidget::setupUI() {
    auto* layout = new QVBoxLayout(this);
    infoLabel_ = new QLabel(this);
    infoLabel_->setWordWrap(true);
    layout->addWidget(infoLabel_);

    auto* refreshBtn = new QPushButton("Refresh", this);
    connect(refreshBtn, &QPushButton::clicked, this, &DatabaseWidget::onRefresh);
    layout->addWidget(refreshBtn);
    layout->addStretch();

    onRefresh();
}

void DatabaseWidget::onRefresh() {
    if (!dbManager_) {
        infoLabel_->setText("Database manager not initialized.");
        return;
    }

    int64_t total = dbManager_->totalCount();
    infoLabel_->setText(QString("Total images: %1\nDatabase: %2\nIndex: %3")
        .arg(total)
        .arg(QString::fromStdString(dbManager_->getDbPath()))
        .arg(QString::fromStdString(dbManager_->getIndexPath())));
}

} // namespace gui
} // namespace vindex
