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
  // Alien glow effect
  float glow =
      sin(v_position.y * 5.0 + v_position.x * 3.0 + v_position.z * 4.0) * 0.1;
  vec3 baseColor = v_color + vec3(glow);

  // Ambient with alien tint
  vec3 ambient = ambient_strength * light_color * vec3(0.8, 1.0, 0.9);

  // Enhanced diffuse for alien surface
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  diff = pow(diff, 1.5); // More dramatic lighting
  vec3 diffuse = diff * light_color;

  // Specular with color shift
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  vec3 specular =
      specular_strength * spec * vec3(0.5, 1.0, 0.8); // Green-tinted specular

  vec3 result = (ambient + diffuse) * baseColor + specular;
  fragColor = vec4(result, 1.0);
}