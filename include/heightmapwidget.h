#pragma once

#include "simulationConfig.h"
#include <QGLBuffer>
#include <QGLShaderProgram>
#include <QGLWidget>
#include <QKeyEvent>
#include <QMatrix4x4>
#include <QVector3D>
#include <vector>

class HeightMapWidget : public QGLWidget {
  Q_OBJECT

public:
  explicit HeightMapWidget(QWidget *parent = nullptr);
  ~HeightMapWidget();

  QSize sizeHint() const override { return QSize(1024, 768); }
  QSize minimumSizeHint() const override { return QSize(800, 600); }

  // Vector of height values from terrain layers
  std::vector<float> layer;

protected:
  void initializeGL() override;
  void paintGL() override;
  void resizeGL(int width, int height) override;
  // void keyPressEvent(QKeyEvent *event) override;
  void keyPressEvent(QKeyEvent *event) override {
    switch (event->key()) {
    case Qt::Key_Left:
      rotateBy(-1, 0, 0);
      break;
    case Qt::Key_Right:
      rotateBy(1, 0, 0);
      break;
    case Qt::Key_Up:
      rotateBy(0, 1, 0);
      break;
    case Qt::Key_Down:
      rotateBy(0, -1, 0);
      break;
    }
    updateGL();
    event->accept();
  };
  void mousePressEvent(QMouseEvent *event) override;
  void mouseMoveEvent(QMouseEvent *event) override;
  void wheelEvent(QWheelEvent *event) override;

private:
  void rotateBy(int x, int y, int z);
  void generateTerrain();
  void updateBuffers();

  // Constants

  // Mesh data
  QVector<QVector3D> m_vertices;
  QVector<QVector3D> m_normals;
  QVector<GLuint> m_indices;

  // GPU Buffers
  QGLBuffer m_vertexbuffer;
  QGLBuffer m_normalbuffer;
  QGLBuffer m_indicebuffer;

  // Shader program
  QGLShaderProgram m_program;

  // View & rotation settings
  QPoint last_pos;
  float distance;
  float x_rot;
  float y_rot;
  float z_rot;

  // Light settings
  QVector3D m_lightPosition;
  QVector3D m_lightColor;
  float m_ambientStrength;
  float m_specularStrength;

  SimulationConfig config;
};