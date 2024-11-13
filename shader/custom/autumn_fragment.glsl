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
  // Foliage variation
  float variation =
      fract(sin(dot(v_position.xz, vec2(12.9898, 78.233))) * 43758.5453) * 0.1;
  vec3 baseColor = v_color + vec3(variation, variation * 0.5, 0.0);

  // Warm ambient for autumn
  vec3 ambient = ambient_strength * light_color * vec3(1.1, 0.9, 0.8);

  // Soft diffuse
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * light_color;

  // Subtle specular
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16.0);
  vec3 specular = specular_strength * spec * light_color * 0.5;

  vec3 result = (ambient + diffuse) * baseColor + specular;
  fragColor = vec4(result, 1.0);
}