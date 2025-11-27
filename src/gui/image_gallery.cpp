#include "image_gallery.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFrame>
#include <QFileInfo>
#include <QImageReader>

namespace vindex {
namespace gui {

// ==================== ImageCard ====================

ImageCard::ImageCard(int64_t id, const QString& imagePath, float score,
                    const QString& label, QWidget* parent)
    : QWidget(parent)
    , id_(id)
    , imagePath_(imagePath)
    , score_(score)
    , label_(label)
{
    setupUI();
    setCursor(Qt::PointingHandCursor);
}

void ImageCard::setupUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(5, 5, 5, 5);
    layout->setSpacing(3);

    // 图像缩略图
    imageLabel_ = new QLabel(this);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setFixedSize(200, 200);
    imageLabel_->setScaledContents(false);  // 不拉伸，保持宽高比
    imageLabel_->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");

    QPixmap thumbnail = loadThumbnail(imagePath_, 200);
    if (!thumbnail.isNull()) {
        imageLabel_->setPixmap(thumbnail);
    } else {
        imageLabel_->setText("No Image");
    }

    layout->addWidget(imageLabel_);

    // 相似度分数
    scoreLabel_ = new QLabel(QString("Score: %1%").arg(score_ * 100, 0, 'f', 1), this);
    scoreLabel_->setAlignment(Qt::AlignCenter);
    scoreLabel_->setStyleSheet("QLabel { color: #0066cc; font-weight: bold; }");
    layout->addWidget(scoreLabel_);

    // 附加信息
    if (!label_.isEmpty()) {
        infoLabel_ = new QLabel(label_, this);
        infoLabel_->setAlignment(Qt::AlignCenter);
        infoLabel_->setWordWrap(true);
        infoLabel_->setStyleSheet("QLabel { color: #666; font-size: 11px; }");
        layout->addWidget(infoLabel_);
    }

    // 设置样式
    setStyleSheet(R"(
        ImageCard {
            background-color: white;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
        }
        ImageCard:hover {
            border-color: #0066cc;
        }
    )");
}

QPixmap ImageCard::loadThumbnail(const QString& path, int size) {
    QImageReader reader(path);

    // 获取原始尺寸，保持宽高比缩放
    QSize originalSize = reader.size();
    if (originalSize.isValid()) {
        QSize scaledSize = originalSize.scaled(size, size, Qt::KeepAspectRatio);
        reader.setScaledSize(scaledSize);
    } else {
        reader.setScaledSize(QSize(size, size));
    }

    QImage image = reader.read();
    if (image.isNull()) {
        return QPixmap();
    }

    return QPixmap::fromImage(image);
}

void ImageCard::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit clicked(id_);
    }
    QWidget::mousePressEvent(event);
}

void ImageCard::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        emit doubleClicked(id_);
    }
    QWidget::mouseDoubleClickEvent(event);
}

void ImageCard::enterEvent(QEnterEvent* event) {
    setStyleSheet(R"(
        ImageCard {
            background-color: #f9f9f9;
            border: 2px solid #0066cc;
            border-radius: 5px;
        }
    )");
    QWidget::enterEvent(event);
}

void ImageCard::leaveEvent(QEvent* event) {
    setStyleSheet(R"(
        ImageCard {
            background-color: white;
            border: 2px solid #e0e0e0;
            border-radius: 5px;
        }
    )");
    QWidget::leaveEvent(event);
}

// ==================== ImageGallery ====================

ImageGallery::ImageGallery(QWidget* parent)
    : QWidget(parent)
    , columns_(4)
    , thumbnailSize_(200)
{
    setupUI();
}

void ImageGallery::setupUI() {
    auto* mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);

    // 创建滚动区域
    scrollArea_ = new QScrollArea(this);
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    // 内容容器
    contentWidget_ = new QWidget();
    gridLayout_ = new QGridLayout(contentWidget_);
    gridLayout_->setSpacing(10);
    gridLayout_->setContentsMargins(10, 10, 10, 10);
    gridLayout_->setAlignment(Qt::AlignTop | Qt::AlignLeft);

    scrollArea_->setWidget(contentWidget_);
    mainLayout->addWidget(scrollArea_);
}

void ImageGallery::setResults(const std::vector<GalleryItem>& items) {
    // 清空现有内容
    clear();

    items_ = items;

    // 创建卡片
    for (size_t i = 0; i < items_.size(); ++i) {
        const auto& item = items_[i];

        auto* card = new ImageCard(
            item.id,
            item.imagePath,
            item.score,
            item.label,
            contentWidget_
        );

        // 连接信号
        connect(card, &ImageCard::clicked,
                this, &ImageGallery::onCardClicked);
        connect(card, &ImageCard::doubleClicked,
                this, &ImageGallery::onCardDoubleClicked);

        cards_.push_back(card);
    }

    // 刷新布局
    refreshLayout();
}

void ImageGallery::clear() {
    // 删除所有卡片
    for (auto* card : cards_) {
        gridLayout_->removeWidget(card);
        card->deleteLater();
    }

    cards_.clear();
    items_.clear();
}

void ImageGallery::setColumns(int columns) {
    if (columns > 0 && columns != columns_) {
        columns_ = columns;
        refreshLayout();
    }
}

void ImageGallery::setThumbnailSize(int size) {
    thumbnailSize_ = size;
}

void ImageGallery::refreshLayout() {
    // 重新排列卡片
    for (size_t i = 0; i < cards_.size(); ++i) {
        int row = i / columns_;
        int col = i % columns_;
        gridLayout_->addWidget(cards_[i], row, col);
    }
}

void ImageGallery::onCardClicked(int64_t id) {
    emit itemClicked(id);
}

void ImageGallery::onCardDoubleClicked(int64_t id) {
    emit itemDoubleClicked(id);
}

} // namespace gui
} // namespace vindex
