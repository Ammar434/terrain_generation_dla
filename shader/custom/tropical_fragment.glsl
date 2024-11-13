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
  // Water shimmer effect for low areas
  float shimmer =
      sin(v_position.x * 4.0 + v_position.z * 4.0 + v_position.y * 10.0) * 0.05;
  vec3 baseColor = v_color;
  if (v_position.y < 0.2) {
    baseColor += vec3(0.0, 0.0, shimmer);
  }

  // Bright ambient for tropical setting
  vec3 ambient = ambient_strength * light_color * vec3(1.0, 1.0, 0.9);

  // Soft diffuse lighting
  vec3 norm = normalize(v_normal);
  vec3 lightDir = normalize(light_position - v_position);
  float diff = max(dot(norm, lightDir), 0.0);
  diff = smoothstep(0.0, 1.0, diff); // Softer light transition
  vec3 diffuse = diff * light_color;

  // Water/foliage specular
  vec3 viewDir = normalize(-v_position);
  vec3 reflectDir = reflect(-lightDir, norm);
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
  vec3 specular = specular_strength * spec * light_color;

  vec3 result = (ambient + diffuse) * baseColor + specular;
  fragColor = vec4(result, 1.0);
}