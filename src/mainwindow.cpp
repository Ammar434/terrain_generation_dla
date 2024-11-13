#include "mainwindow.h"
#include "terrainLayer.h"
#include "ui_mainwindow.h"
#include <QApplication>
#include <QTimer>
#include <heightmapwidget.h>
#include <qopenglwidget.h>
MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);

  SimulationConfig config;

  if (config.buildYourOwnHeightmap) {
    simulation = std::make_unique<Simulation>(config, this);
    TerrainLayers terrainLayers(config.width, config.height, config);
    simulation->runSimulation();
    terrainLayers.processLayers(simulation->getHeightMap());
    terrainLayers.getCombinedLayers();
    HeightMapWidget *hmw = new HeightMapWidget(this);
    hmw->layer = terrainLayers.getCombinedLayers();
    hmw->setMinimumSize(800, 600);
    ui->verticalLayout->addWidget(hmw);
  }

  else {
    if (config.heightMapPath.isEmpty()) {

      qDebug() << "Enter a path for the heightMap\n";
      QTimer::singleShot(0, qApp, &QApplication::quit);
    }
    HeightMapWidget *hmw = new HeightMapWidget(this);
    hmw->setMinimumSize(800, 600);
    ui->verticalLayout->addWidget(hmw);
  }
}

MainWindow::~MainWindow() { delete ui; }
