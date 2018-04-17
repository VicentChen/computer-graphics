#version 330 core

#define LIGHT_NUM 4

in vec3 normal;
in vec2 tex_coord;
in vec3 fragment_pos;

uniform sampler2D texture;
uniform bool blinn;
uniform vec3 light_pos[LIGHT_NUM];
uniform vec3 view_pos;

out vec4 fragment_color;

vec3 calc_light(vec3 color, int i) {

  // ambient
  vec3 ambient = 0.05f * color;

  // diffuse
  vec3 light_dir = normalize(light_pos[i] - fragment_pos);
  float diff = max(dot(light_dir, normal), 0.0f);
  vec3 diffuse = diff * color;

  // specular
  vec3 view_dir = normalize(view_pos - fragment_pos);
  float spec = 0.0f;
  if (blinn) {
    vec3 halfway_dir = normalize(light_dir + view_dir);
    spec = pow(max(dot(normal, halfway_dir), 0.0f), 32.0f);
  }
  else {
    vec3 reflect_dir = reflect(-light_dir, normal);
    spec = pow(max(dot(view_dir, reflect_dir), 0.0f), 8.0f);
  }

  vec3 specular = vec3(0.3f) * spec;

  return ambient + diffuse + specular;
}

void main() {
  vec3 color = texture(texture, tex_coord).rgb;
  vec3 result_color = vec3(0, 0, 0);
  for (int i = 0; i < LIGHT_NUM; i++) {
    result_color += calc_light(color, i);
  }
  fragment_color = vec4(pow(result_color, vec3(1/2.2f)), 1.0f);
}