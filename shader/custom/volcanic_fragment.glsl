#version 450
in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;

uniform vec3 light_position;
uniform vec3 light_color;
uniform float ambient_strength;
uniform float specular_strength;

out vec4 fragColor;

void main() {
  // Lava glow effect
  float glow = max(0.0, 1.0 - v_position.y) * 0.5;
  vec3 baseColor = v_color + vec3(glow, glow * 0.5, 0.0);

  // Warm ambient for lava
  vec3 ambient = ambient_strength * light_color * vec3(1.2, 0.9, 0.7);

  // Diffuse with enhanced contrast
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  diff = pow(diff, 1.2);
  vec3 diffuse = diff * light_color;

  // Hot rock specular
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
  vec3 specular = specular_strength * spec * vec3(1.0, 0.7, 0.3);

  vec3 result = (ambient + diffuse) * baseColor + specular;
  result += vec3(0.3, 0.1, 0.0) * glow; // Additional lava glow
  fragColor = vec4(result, 1.0);
}