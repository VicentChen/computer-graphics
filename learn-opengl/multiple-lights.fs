#version 330 core
struct Material {
  sampler2D diffuse;
  sampler2D specular;
  float shininess;
};

struct DirLight {
  vec3 direction;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct PointLight {
  vec3 position;

  float constant;
  float linear;
  float quadratic;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

struct SpotLight {
  vec3 position;
  vec3 direction;

  float cut_off;
  float outer_cut_off;

  float constant;
  float linear;
  float quadratic;

  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

#define POINT_LIGHT_NUM 4

in vec2 tex_coord;
in vec3 norm;
in vec3 fragment_pos;

out vec4 fragment_color;

uniform vec3 view_pos;
uniform Material material;

uniform DirLight dir_light;
uniform PointLight point_light[POINT_LIGHT_NUM];
uniform SpotLight spot_light;

vec3 calc_dir_light(DirLight light, vec3 norm, vec3 view_dir) {
  vec3 light_dir = normalize(-light.direction);
  // diffuse
  float diff = max(dot(norm, light_dir), 0.0);
  // specular
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);

  vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coord));
  vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coord));
  vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coord));
  return (ambient + diffuse + specular);
}

vec3 calc_point_light(PointLight light, vec3 norm, vec3 fragment_pos, vec3 view_dir) {
  vec3 light_dir = normalize(light.position - fragment_pos);
  // diffuse
  float diff = max(dot(norm, light_dir), 0.0);
  // specular
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);

  // attenuation
  float distance = length(light.position - fragment_pos);
  float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);

  vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coord)) *attenuation;
  vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coord)) * attenuation;
  vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coord)) * attenuation;
  return (ambient + diffuse + specular);
}

vec3 calc_spot_light(SpotLight light, vec3 norm, vec3 fragment_pos, vec3 view_dir) {
  vec3 light_dir = normalize(-light.direction);
  // diffuse
  float diff = max(dot(norm, light_dir), 0.0);
  // specular
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0), material.shininess);

  // attenuation
  float distance = length(light.position - fragment_pos);
  float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * distance * distance);

  // intensity
  float theta = dot(light_dir, normalize(-light.direction));
  float epsilon = light.cut_off - light.outer_cut_off;
  float intensity = clamp((theta - light.outer_cut_off) / epsilon, 0.0, 1.0);

  vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coord)) * attenuation * intensity;
  vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, tex_coord)) * attenuation * intensity;
  vec3 specular = light.specular * spec * vec3(texture(material.specular, tex_coord)) * attenuation * intensity;
  return (ambient + diffuse + specular);
}

void main() {

  vec3 view_dir = normalize(view_pos - fragment_pos);
  vec3 result = calc_dir_light(dir_light, norm, view_dir);

  for(int i = 0; i < POINT_LIGHT_NUM; i++)
    result += calc_point_light(point_light[i], norm, fragment_pos, view_dir);

  result += calc_spot_light(spot_light, norm, fragment_pos, view_dir);
  fragment_color = vec4(result, 1.0f);
}