#version 330 core

in vec3 norm;
in vec3 fragment_pos;
in vec2 tex_coord;

out vec4 fragment_color;

uniform sampler2D texture;
uniform vec3 light_pos;
uniform vec3 view_pos;
uniform vec3 lamp_color;
uniform bool blinn;

void main() {
  vec3 cube_color = vec3(texture(texture, tex_coord));
  vec3 result_color;

  // ambient: Ka * L
  float ambient_strength = 0.1f;
  vec3 ambient = ambient_strength * lamp_color;

  // diffuse: Kd * L
  vec3 light_dir = normalize(light_pos - fragment_pos);
  float diff = max(dot(norm, light_dir), 0.0f);
  vec3 diffuse = diff * lamp_color;

  // specular: Ks * L
  vec3 view_dir = normalize(view_pos - fragment_pos);
  float spec = 0.0f;
  if (blinn) {
    vec3 halfway = normalize(light_dir + view_dir);
    spec = pow(max(dot(norm, halfway), 0.0f), 32.0f);
  }
  else {
    vec3 reflect_dir = reflect(-light_dir, norm);
    spec = pow(max(dot(view_dir, reflect_dir), 0.0f), 8.0f);
  }
  float specular_strength = 0.5f;
  vec3 specular = specular_strength * spec * lamp_color;

  result_color = (ambient + diffuse + specular) * cube_color;
  fragment_color = vec4(result_color, 1.0f);
}
