#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "simulation.h"
#include <QDateTime>
#include <QDir>
#include <QKeyEvent>
#include <QMainWindow>
#include <QOpenGLWidget>
#include <QScreen>
#include <iostream>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

protected:
  // void keyPressEvent(QKeyEvent *event) override;

private:
  Ui::MainWindow *ui;
  std::unique_ptr<Simulation> simulation;

  // void saveScreenshot();
};
#endif // MAINWINDOW_H
