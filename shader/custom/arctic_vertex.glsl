#version 450
in vec3 vertex;
in vec3 normal;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
uniform mat4 matrixpmv;

// Function to get cold-themed color based on height
vec3 getHeightColor(float height) {
  if (height < 0.15) {
    return vec3(0.65, 0.75, 0.85); // Ice blue for lowest areas
  } else if (height < 0.3) {
    return vec3(0.8, 0.85, 0.9); // Light blue-white for low areas
  } else if (height < 0.5) {
    return vec3(0.9, 0.9, 0.95); // Almost white snow
  } else if (height < 0.7) {
    return vec3(0.82, 0.87, 0.89); // Slightly blue tinted snow
  } else if (height < 0.85) {
    return vec3(0.7, 0.7, 0.75); // Grey rock with snow
  } else {
    return vec3(0.95, 0.95, 1.0); // Pure snow caps
  }
}

void main() {
  v_normal = normal;
  v_position = vertex;

  // Calculate color based on height
  v_color = getHeightColor(vertex.y / 2.0);

  gl_Position = matrixpmv * vec4(vertex, 1.0);
}