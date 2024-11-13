#version 450
in vec3 v_normal;
in vec3 v_position;
in vec3 v_color;

uniform vec3 light_position;
uniform vec3 light_color;
uniform float ambient_strength;
uniform float specular_strength;

out vec4 fragColor;

// Noise function for snow texture variation
float rand(vec2 co) {
  return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
  // Add slight color variation for snow texture
  float noise = rand(v_position.xz) * 0.1;
  vec3 baseColor = v_color + vec3(noise);

  // Ambient with cold blue tint
  vec3 ambient = (ambient_strength + 0.1) * light_color * vec3(0.9, 0.95, 1.0);

  // Diffuse
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * light_color;

  // Enhanced specular for icy shine
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec =
      pow(max(dot(viewDir, reflectDir), 0.0), 64.0); // Higher shininess
  vec3 specular =
      specular_strength * 1.5 * spec * light_color; // Increased specular

  // Combine lighting with snow color
  vec3 result = (ambient + diffuse) * baseColor + specular;

  // Add slight blue tint to final color
  result = mix(result, result * vec3(0.95, 0.97, 1.0), 0.2);

  fragColor = vec4(result, 1.0);
}