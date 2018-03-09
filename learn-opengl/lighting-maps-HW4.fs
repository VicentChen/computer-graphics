#version 330 core
struct Material {
  // vec3 ambient;
  sampler2D diffuse;
  sampler2D specular;
  sampler2D emission;
  float shininess;
};

struct Light {
  vec3 position;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

in vec3 norm;
in vec3 fragment_pos;
in vec2 tex_coord;

out vec4 fragment_color;

uniform vec3 view_pos;

uniform Material material;
uniform Light light;

void main() {
  vec3 result_color;

  // ambient: Ka * L
  vec3 ambient = light.ambient * vec3(texture(material.diffuse, tex_coord));

  // diffuse: Kd * L
  vec3 light_dir = normalize(light.position - fragment_pos);
  float diff = max(dot(norm, light_dir), 0.0f);
  vec3 diffuse = light.diffuse * (diff * vec3(texture(material.diffuse, tex_coord)));

  // specular: Ks * L
  vec3 view_dir = normalize(view_pos - fragment_pos);
  vec3 reflect_dir = reflect(-light_dir, norm);
  float spec = pow(max(dot(view_dir, reflect_dir), 0.0f), material.shininess);
  vec3 specular = light.specular * (spec * vec3(texture(material.specular, tex_coord)));

  // emission
  vec3 emission = vec3(texture(material.emission, tex_coord));

  result_color = ambient + diffuse + specular + emission;
  fragment_color = vec4(result_color, 1.0f);
}
