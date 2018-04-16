#version 330 core

in vec3 normal;
in vec2 tex_coord;
in vec3 fragment_pos;

uniform sampler2D texture;
uniform bool blinn;
uniform vec3 light_pos;
uniform vec3 view_pos;

out vec4 fragment_color;

void main() {
  vec3 color = texture(texture, tex_coord).rgb;
  
  // ambient
  vec3 ambient = 0.05f * color;

  // diffuse
  vec3 light_dir = normalize(light_pos - fragment_pos);
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

  vec3 result_color = ambient + diffuse + specular;
  fragment_color = vec4(result_color, 1.0f);
}