#version 330 core

in vec3 normal;
in vec2 tex_coord;
in vec3 fragment_pos;
in vec4 fragment_pos_lightspace;

uniform sampler2D texture;
uniform sampler2D shadow_map;
uniform bool blinn;
uniform vec3 light_pos;
uniform vec3 view_pos;

out vec4 fragment_color;

float calc_shadow(vec4 fragment_pos_lightspace, vec3 light_dir) {
  vec3 proj_coord = fragment_pos_lightspace.xyz / fragment_pos_lightspace.w;
  vec2 texel_size = 1.0f / textureSize(shadow_map, 0);
  proj_coord = proj_coord * 0.5 + 0.5;
  if (proj_coord.z > 1.0f) return 0.0f;
  float shadow = 0.0f;
  for (int x = -1; x <= 1; x++)
    for (int y = -1; y <= 1; y++) {
      float closest_depth = texture(shadow_map, proj_coord.xy + vec2(x,y) * texel_size).r;
      float current_depth = proj_coord.z;
      float bias = max(0.05 * (1 - dot(normal, light_dir)), 0.005);
      shadow += current_depth - bias > closest_depth ? 1.0f : 0.0f;
    }
  return shadow / 9.0f;
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
  float shadow = calc_shadow(fragment_pos_lightspace, light_dir);

  return (ambient + (1-shadow) * (diffuse + specular)) * color;
  // return shadow;
}

void main() {
  vec3 color = texture(texture, tex_coord).rgb;
  vec3 result_color = vec3(0, 0, 0);
  result_color += calc_light(color);
  fragment_color = vec4(result_color, 1.0f);
}