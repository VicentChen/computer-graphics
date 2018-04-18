#version 330 core

in vec3 normal;
in vec2 tex_coord;
in vec3 fragment_pos;

uniform sampler2D texture;
uniform samplerCube shadow_map;
uniform bool blinn;
uniform vec3 light_pos;
uniform vec3 view_pos;
uniform float far_plane;
uniform bool shadows;

out vec4 fragment_color;

float calc_shadow(vec3 fragment_pos) {
  vec3 fragToLight = fragment_pos - light_pos;
  float closestDepth = texture(shadow_map, fragToLight).r;
  closestDepth *= far_plane;
  float currentDepth = length(fragToLight);
  float bias = 0.05;
  float shadow = currentDepth -  bias > closestDepth ? 1.0 : 0.0;   
  return shadow;
}

vec3 calc_light(vec3 color) {
  vec3 light_color = vec3(1.0f);

  // ambient
  vec3 ambient = 0.3f * color;

  // diffuse
  vec3 light_dir = normalize(light_pos - fragment_pos);
  float diff = max(dot(light_dir, normal), 0.0f);
  vec3 diffuse = diff * light_color;

  // specular
  vec3 view_dir = normalize(view_pos - fragment_pos);
  float spec = 0.0f;
  if (blinn) {
    vec3 halfway_dir = normalize(light_dir + view_dir);
    spec = pow(max(dot(normal, halfway_dir), 0.0f), 64.0f);
  }
  else {
    vec3 reflect_dir = reflect(-light_dir, normal);
    spec = pow(max(dot(view_dir, reflect_dir), 0.0f), 8.0f);
  }
  vec3 specular = spec * light_color;

  // shadow
  float shadow = shadows ? calc_shadow(fragment_pos) : 0.0f;

  return (ambient + (1-shadow) * (diffuse + specular)) * color;
}

void main() {
  vec3 color = texture(texture, tex_coord).rgb;
  vec3 result_color = vec3(0, 0, 0);
  result_color += calc_light(color);
  fragment_color = vec4(result_color, 1.0f);
}