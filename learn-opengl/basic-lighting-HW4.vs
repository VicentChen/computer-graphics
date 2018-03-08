#version 330 core

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;

out vec4 vertex_color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 light_pos;
uniform vec3 view_pos;
uniform vec3 lamp_color;
uniform vec3 cube_color;

void main() {
  // position
  gl_Position = projection * view * model * vec4(pos, 1.0f);

  // color(world space)
  vec3 norm = normalize(mat3(transpose(inverse(model))) * normal);
  
  // ambient: Ka * L
  float ambient_strength = 0.1f;
  vec3 ambient = ambient_strength * lamp_color;

  // diffuse: Kd * L
  vec3 vertex_pos = vec3(model * vec4(pos, 1.0f));
  vec3 light_dir = normalize(light_pos - vertex_pos);
  float diff = max(dot(norm, light_dir), 0.0f);
  vec3 diffuse = diff * lamp_color;

  // specular: Ks * L
  vec3 view_dir = normalize(view_pos - vertex_pos);
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), 32);
  float specular_strength = 0.5f;
  vec3 specular = specular_strength * spec * lamp_color;

  vec3 result_color = (ambient + diffuse + specular) * cube_color;
  vertex_color = vec4(result_color, 1.0f);
}