#version 450

in vec3 vertex;
in vec3 normal;
out vec3 v_normal;
out vec3 v_position;
out vec3 v_color;
uniform mat4 matrixpmv;

// Helper function to smoothly interpolate between colors
vec3 smoothInterpolate(vec3 color1, vec3 color2, float value, float min,
                       float max) {
  float t = (value - min) / (max - min);
  t = clamp(t, 0.0, 1.0);
  // Smooth step for more natural transition
  t = t * t * (3.0 - 2.0 * t);
  return mix(color1, color2, t);
}

vec3 getHeightColor(float height) {
  // Base colors for different terrain types
  vec3 lowValley = vec3(0.87, 0.78, 0.60); // Light sandy base
  vec3 dunes = vec3(0.83, 0.69, 0.51);     // Medium sand dunes
  vec3 highDunes = vec3(0.76, 0.60, 0.42); // Darker sand ridges
  vec3 rockBase = vec3(0.71, 0.52, 0.35);  // Rocky outcrops
  vec3 rockTop = vec3(0.62, 0.45, 0.32);   // Mountain rock

  // Smooth transitions between different height levels
  if (height < 0.2) {
    return lowValley;
  } else if (height < 0.4) {
    return smoothInterpolate(lowValley, dunes, height, 0.2, 0.4);
  } else if (height < 0.6) {
    return smoothInterpolate(dunes, highDunes, height, 0.4, 0.6);
  } else if (height < 0.8) {
    return smoothInterpolate(highDunes, rockBase, height, 0.6, 0.8);
  } else {
    return smoothInterpolate(rockBase, rockTop, height, 0.8, 1.0);
  }
}

void main() {
  v_normal = normal;
  v_position = vertex;

  // Calculate color based on normalized height
  v_color = getHeightColor(vertex.y / 2.0);

  gl_Position = matrixpmv * vec4(vertex, 1.0);
}