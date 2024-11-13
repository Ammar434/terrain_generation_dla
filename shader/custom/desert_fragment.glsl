#version 450
in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;

uniform vec3 light_position;
uniform vec3 light_color;
uniform float ambient_strength;
uniform float specular_strength;

out vec4 fragColor;

// Function to add slight color variation
vec3 addVariation(vec3 color, vec3 position) {
  float variation = sin(position.x * 10.0) * cos(position.z * 10.0) * 0.03;
  return color * (1.0 + variation);
}

void main() {
  // Base ambient lighting for desert environment
  vec3 ambient = (ambient_strength + 0.15) * light_color;

  // Enhanced diffuse lighting for terrain detail
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  // Enhance shadows slightly
  diff = pow(diff, 1.1);
  vec3 diffuse = diff * light_color;

  // Subtle specular for rocky areas
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  vec3 specular = specular_strength * spec * light_color * 0.2;

  // Add subtle color variation based on position
  vec3 baseColor = addVariation(v_color, v_position);

  // Combine all lighting components
  vec3 result = (ambient + diffuse + specular) * baseColor;

  // Add slight atmospheric depth
  float depth = (v_position.z + 25.0) / 50.0; // Normalize to 0-1 range
  depth = clamp(depth, 0.0, 1.0);
  vec3 atmosphericColor = vec3(0.80, 0.70, 0.60);
  result = mix(atmosphericColor, result, depth);

  // Adjust final color temperature for desert lighting
  result = mix(result, result * vec3(1.05, 1.02, 0.98), 0.2);

  fragColor = vec4(result, 1.0);
}