#pragma once

#include <QScrollArea>
#include <QGridLayout>
#include <QLabel>
#include <QPixmap>
#include <QPushButton>
#include <QWidget>
#include <QMouseEvent>
#include <vector>
#include <string>

namespace vindex {
namespace gui {

/**
 * @brief 单个图像项卡片
 */
class ImageCard : public QWidget {
    Q_OBJECT

public:
    ImageCard(int64_t id, const QString& imagePath, float score,
              const QString& label, QWidget* parent = nullptr);

    int64_t getId() const { return id_; }
    QString getImagePath() const { return imagePath_; }

signals:
    void clicked(int64_t id);
    void doubleClicked(int64_t id);

protected:
    void mousePressEvent(QMouseEvent* event) override;
    void mouseDoubleClickEvent(QMouseEvent* event) override;
    void enterEvent(QEnterEvent* event) override;
    void leaveEvent(QEvent* event) override;

private:
    void setupUI();
    QPixmap loadThumbnail(const QString& path, int size);

private:
    int64_t id_;
    QString imagePath_;
    float score_;
    QString label_;

    QLabel* imageLabel_;
    QLabel* scoreLabel_;
    QLabel* infoLabel_;
};

/**
 * @brief 图像结果展示组件
 *
 * 网格布局展示搜索结果
 * 支持：
 * - 可调列数
 * - 缩略图展示
 * - 点击和双击事件
 * - 相似度分数显示
 */
class ImageGallery : public QWidget {
    Q_OBJECT

public:
    /**
     * @brief 结果项
     */
    struct GalleryItem {
        int64_t id;
        QString imagePath;
        float score;
        QString label;

        GalleryItem(int64_t i, const QString& path, float s, const QString& l = "")
            : id(i), imagePath(path), score(s), label(l) {}
    };

    explicit ImageGallery(QWidget* parent = nullptr);
    ~ImageGallery() = default;

    /**
     * @brief 设置结果
     * @param items 结果项列表
     */
    void setResults(const std::vector<GalleryItem>& items);

    /**
     * @brief 清空结果
     */
    void clear();

    /**
     * @brief 设置列数
     */
    void setColumns(int columns);

    /**
     * @brief 设置缩略图尺寸
     */
    void setThumbnailSize(int size);

    /**
     * @brief 获取结果数量
     */
    int getResultCount() const { return items_.size(); }

signals:
    void itemClicked(int64_t id);
    void itemDoubleClicked(int64_t id);

private slots:
    void onCardClicked(int64_t id);
    void onCardDoubleClicked(int64_t id);

private:
    void setupUI();
    void refreshLayout();

private:
    QScrollArea* scrollArea_;
    QWidget* contentWidget_;
    QGridLayout* gridLayout_;

    std::vector<GalleryItem> items_;
    std::vector<ImageCard*> cards_;

    int columns_;
    int thumbnailSize_;
};

} // namespace gui
} // namespace vindex
