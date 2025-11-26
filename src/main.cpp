#include <QApplication>
#include <QStyleFactory>
#include <iostream>
#include "gui/main_window.h"

int main(int argc, char* argv[]) {
    // 创建Qt应用
    QApplication app(argc, argv);

    // 设置应用信息
    QApplication::setApplicationName("VIndex");
    QApplication::setApplicationVersion("1.0.0");
    QApplication::setOrganizationName("VIndex");
    QApplication::setOrganizationDomain("vindex.local");

    // 设置应用样式
    QApplication::setStyle(QStyleFactory::create("Fusion"));

    // 设置调色板（可选：浅色主题）
    QPalette palette;
    palette.setColor(QPalette::Window, QColor(240, 240, 240));
    palette.setColor(QPalette::WindowText, Qt::black);
    palette.setColor(QPalette::Base, Qt::white);
    palette.setColor(QPalette::AlternateBase, QColor(245, 245, 245));
    palette.setColor(QPalette::ToolTipBase, Qt::white);
    palette.setColor(QPalette::ToolTipText, Qt::black);
    palette.setColor(QPalette::Text, Qt::black);
    palette.setColor(QPalette::Button, QColor(240, 240, 240));
    palette.setColor(QPalette::ButtonText, Qt::black);
    palette.setColor(QPalette::Link, QColor(0, 102, 204));
    palette.setColor(QPalette::Highlight, QColor(0, 102, 204));
    palette.setColor(QPalette::HighlightedText, Qt::white);
    QApplication::setPalette(palette);

    try {
        // 创建并显示主窗口
        vindex::gui::MainWindow mainWindow;
        mainWindow.show();

        // 运行应用
        return app.exec();

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
