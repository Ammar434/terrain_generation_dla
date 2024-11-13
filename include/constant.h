#ifndef CONSTANT_H
#define CONSTANT_H
static constexpr float PARTICLE_RADIUS = 2.0f;
static constexpr float MOVE_RADIUS = PARTICLE_RADIUS * 0.5;
static constexpr float SPAWN_MARGIN = 50.0f;

static constexpr int SIMULATION_WIDTH = 1024 - SPAWN_MARGIN;
static constexpr int SIMULATION_HEIGHT = 1024 - SPAWN_MARGIN;
static constexpr int NUM_PARTICLES = 10000;

#endif