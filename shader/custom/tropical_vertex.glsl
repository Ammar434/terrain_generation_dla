#version 450
in vec3 vertex;
in vec3 normal;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
uniform mat4 matrixpmv;

vec3 getHeightColor(float height) {
  if (height < 0.15) {
    return vec3(0.0, 0.7, 0.9); // Turquoise water
  } else if (height < 0.25) {
    return vec3(0.9, 0.9, 0.7); // Sandy beach
  } else if (height < 0.5) {
    return vec3(0.0, 0.8, 0.2); // Tropical vegetation
  } else if (height < 0.7) {
    return vec3(0.0, 0.6, 0.1); // Dense jungle
  } else if (height < 0.85) {
    return vec3(0.5, 0.4, 0.3); // Mountain rock
  } else {
    return vec3(0.7, 0.7, 0.7); // Rocky peaks
  }
}

void main() {
  v_normal = normal;
  v_position = vertex;
  v_color = getHeightColor(vertex.y / 2.0);
  gl_Position = matrixpmv * vec4(vertex, 1.0);
}