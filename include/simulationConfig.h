#pragma once
#include <QColor>
#include <QString>
#include <QVector3D>

struct SimulationConfig {
  // Basic simulation parameters
  int width = 1080;
  int height = 1080;
  int numParticles = 15000;
  float particleRadius = 2.0f;
  float moveRadius = 1.0f;
  float spawnMargin = 10.0f;

  // Terrain generation parameters
  float initialRadius = 1.5f; // smoothing
  int numLayers = 6;          // layers of detail for blur
  int initialFrozenParticles = 1;

  // Directory to save images
  QString outputDirectory = "heightmap/one_frozen";
  bool buildYourOwnHeightmap = false;

  // Desert Theme (default)
  // QString heightMapPath =
  //     "./heightmap/five_frozen/combined_2024-11-13_01-20-11.png";
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(1.0f, 0.9f, 0.8f);
  // float m_ambientStrength = 0.3f;
  // float m_specularStrength = 0.1f;
  // QColor backgroundColor = QColor(60, 70, 110);
  // QString vertexPath = "./shader/custom/desert_vertex.glsl";
  // QString fragmentPath = "./shader/custom/desert_fragment.glsl";

  // Arctic Theme - uncomment to use
  // QString heightMapPath =
  //     "./heightmap/one_frozen/combined_2024-11-13_04-53-44.png";
  // QColor backgroundColor = QColor(70, 60, 90);
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(0.95f, 0.95f, 1.0f);
  // float m_ambientStrength = 0.35f;
  // float m_specularStrength = 0.8f;
  // QString vertexPath = "./shader/custom/arctic_vertex.glsl";
  // QString fragmentPath = "./shader/custom/arctic_fragment.glsl";

  // Alien Theme - uncomment to use
  // QString heightMapPath =
  //     "./heightmap/three_frozen/combined_2024-11-13_02-33-50.png";
  // QColor backgroundColor = QColor(255, 180, 50);
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(0.8f, 0.9f, 1.0f);
  // float m_ambientStrength = 0.3f;
  // float m_specularStrength = 0.6f;
  // QString vertexPath = "./shader/custom/alien_vertex.glsl";
  // QString fragmentPath = "./shader/custom/alien_fragment.glsl";

  // Volcanic Theme - uncomment to use
  QString heightMapPath =
      "./heightmap/five_frozen/combined_2024-11-13_02-09-20.png";
  QColor backgroundColor = QColor(100, 140, 180);
  QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  QVector3D m_lightColor = QVector3D(1.0f, 0.9f, 0.8f);
  float m_ambientStrength = 0.2f;
  float m_specularStrength = 0.4f;
  QString vertexPath = "./shader/custom/volcanic_vertex.glsl";
  QString fragmentPath = "./shader/custom/volcanic_fragment.glsl";

  // Tropical Theme - uncomment to use
  // QString heightMapPath =
  //     "./heightmap/three_frozen/combined_2024-11-13_02-37-33.png";
  // QColor backgroundColor = QColor(255, 180, 170);
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(1.0f, 1.0f, 0.95f);
  // float m_ambientStrength = 0.25f;
  // float m_specularStrength = 0.3f;
  // QString vertexPath = "./shader/custom/tropical_vertex.glsl";
  // QString fragmentPath = "./shader/custom/tropical_fragment.glsl";

  // Autumn Theme - uncomment to use
  // QString heightMapPath =
  //     "./heightmap/two_frozen/combined_2024-11-13_03-06-35.png";
  // QColor backgroundColor = QColor(70, 100, 150);
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(1.0f, 0.95f, 0.8f);
  // float m_ambientStrength = 0.4f;
  // float m_specularStrength = 0.2f;
  // QString vertexPath = "./shader/custom/autumn_vertex.glsl";
  // QString fragmentPath = "./shader/custom/autumn_fragment.glsl";

  // Mars Theme - uncomment to use
  // QString heightMapPath =
  //     "./heightmap/two_frozen/combined_2024-11-13_03-27-13.png";
  // QColor backgroundColor = QColor(40, 130, 140);
  // QVector3D m_lightPosition = QVector3D(5.0f, 15.0f, 5.0f);
  // QVector3D m_lightColor = QVector3D(1.0f, 0.9f, 0.8f);
  // float m_ambientStrength = 0.3f;
  // float m_specularStrength = 0.1f;
  // QString vertexPath = "./shader/custom/mars_vertex.glsl";
  // QString fragmentPath = "./shader/custom/mars_fragment.glsl";
};