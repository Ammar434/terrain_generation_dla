#ifndef PARTICLE_H
#define PARTICLE_H

class Particle {
public:
  float x, y;
  float oldX, oldY;
  float elevation;
  bool isActive;
  int frozenAtStep;
  int collidedParticleIdx;
  int iteration;

  Particle()
      : x(0), y(0), oldX(0), oldY(0), elevation(0), isActive(true),
        frozenAtStep(-1), collidedParticleIdx(-1), iteration(0) {}
};
#endif