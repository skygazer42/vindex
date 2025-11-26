#pragma once

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include "../index/database_manager.h"

namespace vindex {
namespace gui {

/**
 * @brief 图库管理占位界面
 *
 * 提供基础信息显示与未来扩展挂载点。
 */
class DatabaseWidget : public QWidget {
    Q_OBJECT
public:
    explicit DatabaseWidget(index::DatabaseManager* dbManager,
                            QWidget* parent = nullptr);
    ~DatabaseWidget() = default;

private slots:
    void onRefresh();

private:
    void setupUI();

private:
    index::DatabaseManager* dbManager_;
    QLabel* infoLabel_;
};

} // namespace gui
} // namespace vindex
