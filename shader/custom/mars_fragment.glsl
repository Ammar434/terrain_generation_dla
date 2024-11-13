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
  // Dust effect
  float dust =
      fract(sin(dot(v_position.xz, vec2(15.9898, 88.233))) * 43758.5453) * 0.08;
  vec3 baseColor = v_color + vec3(dust, dust * 0.8, dust * 0.6);

  // Dim ambient for Mars
  vec3 ambient = ambient_strength * light_color * vec3(1.1, 0.9, 0.8);

  // Harsh diffuse lighting
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  diff = pow(diff, 1.3); // Harsher shadows
  vec3 diffuse = diff * light_color;

  // Dusty specular
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 8.0); // More spread out
  vec3 specular = specular_strength * spec * light_color * 0.3;

  vec3 result = (ambient + diffuse) * baseColor + specular;
  // Add slight red atmospheric tint
  result = mix(result, result * vec3(1.1, 0.9, 0.8), 0.1);
  fragColor = vec4(result, 1.0);
}