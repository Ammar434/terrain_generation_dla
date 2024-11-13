#include "mainwindow.h"
#include <QApplication>
#include <QScreen>

int main(int argc, char *argv[]) {
  QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
  QApplication::setAttribute(Qt::AA_UseHighDpiPixmaps);

  QApplication a(argc, argv);
  MainWindow w;
  QScreen *screen = QGuiApplication::primaryScreen();

  if (screen) {
    QRect screenGeometry = screen->geometry();

    int width = screenGeometry.width() * 0.8;
    int height = screenGeometry.height() * 0.8;

    int x = (screenGeometry.width() - width) / 2;
    int y = (screenGeometry.height() - height) / 2;

    w.resize(width, height);
    w.move(x, y);
  }

  w.show();
  return a.exec();
}
