#version 330 core

in vec3 norm;
in vec3 fragment_pos;

out vec4 fragment_color;

uniform vec3 light_pos;
uniform vec3 view_pos;

uniform vec3 lamp_color;
uniform vec3 cube_color;

void main() {

  // ambient
  float ambient_strength = 0.1f;
  vec3 ambient = ambient_strength * lamp_color;

  // diffuse
  vec3 light_dir = normalize(light_pos - fragment_pos);
  float diff = max(dot(light_dir, norm), 0.0f);
  vec3 diffuse = diff * lamp_color;

  // specular-1
  // vec3 view_dir = normalize(view_pos - fragment_pos);
  // vec3 reflect_dir = reflect(-light_dir, norm);
  // float spec = pow(max(dot(reflect_dir, view_dir), 0.0f), 128);
  // float specular_strength = 0.5f;
  // vec3 specular = specular_strength * spec * lamp_color;

  // specular-2
  vec3 view_dir = normalize(view_pos - fragment_pos);
  vec3 halfway_vec = normalize(light_dir + view_dir);
  float spec = pow(max(dot(halfway_vec, norm), 0.0f), 128);
  float specular_strength = 0.5f;
  vec3 specular = specular_strength * spec * lamp_color;

  vec3 result_color = (ambient + diffuse + specular) * cube_color;
  fragment_color = vec4(result_color, 1.0f);
}
