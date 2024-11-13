#version 450
in vec3 vertex;
in vec3 normal;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
uniform mat4 matrixpmv;

vec3 getHeightColor(float height) {
  if (height < 0.2) {
    return vec3(1.0, 0.2, 0.0); // Lava
  } else if (height < 0.4) {
    return vec3(0.6, 0.1, 0.0); // Cooling lava
  } else if (height < 0.6) {
    return vec3(0.3, 0.3, 0.3); // Dark rock
  } else if (height < 0.8) {
    return vec3(0.4, 0.4, 0.4); // Volcanic rock
  } else {
    return vec3(0.2, 0.2, 0.2); // Ash peaks
  }
}

void main() {
  v_normal = normal;
  v_position = vertex;
  v_color = getHeightColor(vertex.y / 2.0);
  gl_Position = matrixpmv * vec4(vertex, 1.0);
}