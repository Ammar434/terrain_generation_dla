#include "heightmapwidget.h"
#include <QDebug>
#include <QtGui/QMouseEvent>
#include <cmath>

const float MAP_SIZE = 20.0f;

HeightMapWidget::HeightMapWidget(QWidget *parent)
    : QGLWidget(parent), m_vertexbuffer(QGLBuffer::VertexBuffer),
      m_normalbuffer(QGLBuffer::VertexBuffer),
      m_indicebuffer(QGLBuffer::IndexBuffer),
      m_lightPosition(10.0f, 20.0f, 10.0f), m_lightColor(1.0f, 0.97f, 0.94f),
      m_ambientStrength(0.25f), m_specularStrength(0.15f) {

  m_lightPosition = config.m_lightPosition;
  m_lightColor = config.m_lightColor;
  m_ambientStrength = config.m_ambientStrength;
  m_specularStrength = config.m_specularStrength;

  distance = -30.0f;
  x_rot = 30;
  y_rot = 5;
  z_rot = 0;

  setFocusPolicy(Qt::StrongFocus);
  setAttribute(Qt::WA_OpaquePaintEvent);
  setAutoFillBackground(false);
}

HeightMapWidget::~HeightMapWidget() {
  m_vertexbuffer.destroy();
  m_normalbuffer.destroy();
  m_indicebuffer.destroy();
}

void HeightMapWidget::generateTerrain() {
  QImage img;
  int width;
  int height;

  if (config.buildYourOwnHeightmap) {
    width = config.width;
    height = config.height;

  } else {
    img = QImage(config.heightMapPath);
    width = img.width();
    height = img.height();
  }

  qDebug() << "Generating terrain with dimensions:" << width << "x" << height;

  m_vertices.clear();
  m_normals.clear();
  m_indices.clear();

  // Calculate scales maintaining aspect ratio
  float aspectRatio = width / height;
  float zScale = MAP_SIZE / (height - 1);
  float xScale = zScale * aspectRatio;

  // Center offset calculation
  float xOffset = (width * xScale) / 2.0f;
  float zOffset = MAP_SIZE / 2.0f;

  // Generate vertices
  for (int z = 0; z < height; ++z) {
    for (int x = 0; x < width; ++x) {

      if (config.buildYourOwnHeightmap) {

        // Calculate vertex position
        float xPos = x * xScale - xOffset;
        float zPos = z * zScale - zOffset;

        float yPos = layer[z * width + x] * 5.0f;
        m_vertices.append(QVector3D(xPos, yPos, zPos));
        m_normals.append(QVector3D(0, 1, 0));
      } else {
        QRgb color = img.pixel(x, z);

        // Calculate vertex position
        float xPos = x * xScale - xOffset;
        float zPos = z * zScale - zOffset;
        float yPos = 3.0 * qGray(color) / 255.0f;

        m_vertices.append(QVector3D(xPos, yPos, zPos));
        m_normals.append(QVector3D(0, 1, 0));
      }
    }
  }

  // Generate indices for triangle strips
  for (int z = 0; z < height - 1; ++z) {
    for (int x = 0; x < width - 1; ++x) {
      int topLeft = z * width + x;
      int topRight = topLeft + 1;
      int bottomLeft = (z + 1) * width + x;
      int bottomRight = bottomLeft + 1;

      // First triangle
      m_indices.append(topLeft);
      m_indices.append(bottomLeft);
      m_indices.append(topRight);

      // Second triangle
      m_indices.append(topRight);
      m_indices.append(bottomLeft);
      m_indices.append(bottomRight);
    }
  }

  // Calculate normals
  m_normals.fill(QVector3D(0, 0, 0));

  for (int i = 0; i < m_indices.size(); i += 3) {
    QVector3D v1 = m_vertices[m_indices[i]];
    QVector3D v2 = m_vertices[m_indices[i + 1]];
    QVector3D v3 = m_vertices[m_indices[i + 2]];

    QVector3D normal = QVector3D::crossProduct(v2 - v1, v3 - v1).normalized();

    m_normals[m_indices[i]] += normal;
    m_normals[m_indices[i + 1]] += normal;
    m_normals[m_indices[i + 2]] += normal;
  }

  // Normalize all normals
  for (int i = 0; i < m_normals.size(); ++i) {
    m_normals[i].normalize();
  }

  updateBuffers();
}

void HeightMapWidget::updateBuffers() {
  if (!m_vertexbuffer.isCreated() || !m_normalbuffer.isCreated() ||
      !m_indicebuffer.isCreated()) {
    qDebug() << "Error: Buffers not created!";
    return;
  }

  // Update vertex buffer
  m_vertexbuffer.bind();
  m_vertexbuffer.allocate(m_vertices.constData(),
                          m_vertices.size() * sizeof(QVector3D));
  m_vertexbuffer.release();

  // Update normal buffer
  m_normalbuffer.bind();
  m_normalbuffer.allocate(m_normals.constData(),
                          m_normals.size() * sizeof(QVector3D));
  m_normalbuffer.release();

  // Update index buffer
  m_indicebuffer.bind();
  m_indicebuffer.allocate(m_indices.constData(),
                          m_indices.size() * sizeof(GLuint));
  m_indicebuffer.release();
}

void HeightMapWidget::initializeGL() {

  m_vertexbuffer.create();
  m_normalbuffer.create();
  m_indicebuffer.create();

  // Initialize shader program
  if (!m_program.addShaderFromSourceFile(QGLShader::Vertex,
                                         config.vertexPath)) {
    qDebug() << "Failed to load vertex shader:" << m_program.log();
    return;
  }

  if (!m_program.addShaderFromSourceFile(QGLShader::Fragment,
                                         config.fragmentPath)) {
    qDebug() << "Failed to load fragment shader:" << m_program.log();
    return;
  }

  // Bind attribute locations
  m_program.bindAttributeLocation("vertex", 0);
  m_program.bindAttributeLocation("normal", 1);

  if (!m_program.link()) {
    qDebug() << "Shader linking failed:" << m_program.log();
    return;
  }

  // Generate the terrain mesh
  generateTerrain();

  // OpenGL settings
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);

  // qglClearColor(Qt::black);

  qglClearColor(config.backgroundColor);
}

void HeightMapWidget::paintGL() {
  if (m_vertices.isEmpty() || m_indices.isEmpty()) {
    return;
  }

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  m_program.bind();

  // Set up matrices
  QMatrix4x4 projection;
  projection.perspective(45.0f, float(width()) / height(), 0.01f, 100.0f);

  QMatrix4x4 view;
  view.translate(0.0f, -2.0f, distance); // Added Y offset to center terrain

  QMatrix4x4 model;
  model.rotate(x_rot, 1.0f, 0.0f, 0.0f);
  model.rotate(y_rot, 0.0f, 1.0f, 0.0f);
  model.rotate(z_rot, 0.0f, 0.0f, 1.0f);

  // Set uniforms
  m_program.setUniformValue("matrixpmv", projection * view * model);
  m_program.setUniformValue("light_position", m_lightPosition);
  m_program.setUniformValue("light_color", m_lightColor);
  m_program.setUniformValue("ambient_strength", m_ambientStrength);
  m_program.setUniformValue("specular_strength", m_specularStrength);

  // Draw terrain
  m_vertexbuffer.bind();
  m_program.enableAttributeArray(0);
  m_program.setAttributeBuffer(0, GL_FLOAT, 0, 3);
  m_vertexbuffer.release();

  m_normalbuffer.bind();
  m_program.enableAttributeArray(1);
  m_program.setAttributeBuffer(1, GL_FLOAT, 0, 3);
  m_normalbuffer.release();

  m_indicebuffer.bind();
  glDrawElements(GL_TRIANGLES, m_indices.size(), GL_UNSIGNED_INT, nullptr);
  m_indicebuffer.release();

  m_program.release();
}

void HeightMapWidget::mousePressEvent(QMouseEvent *event) {
  last_pos = event->pos();
}

void HeightMapWidget::mouseMoveEvent(QMouseEvent *event) {
  int dx = event->x() - last_pos.x();
  int dy = event->y() - last_pos.y();

  if (event->buttons() & Qt::LeftButton) {
    rotateBy(dy, dx, 0);
  }
  last_pos = event->pos();
  updateGL();
}

void HeightMapWidget::wheelEvent(QWheelEvent *event) {
  QPoint numDegrees = event->angleDelta() / 8;
  float numSteps = numDegrees.y() / 15.0f;
  distance *= 1.0f + (numSteps / 10.0f);
  updateGL();
}

void HeightMapWidget::rotateBy(int x, int y, int z) {
  x_rot += x;
  y_rot += y;
  z_rot += z;
}

void HeightMapWidget::resizeGL(int width, int height) {
  qDebug() << "Widget resized to:" << width << "x" << height;
  glViewport(0, 0, width, height);
}
