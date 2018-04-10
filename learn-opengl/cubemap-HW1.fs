#version 330 core

in vec3 normal;
in vec3 position;
in vec2 tex_coord;

out vec4 fragment_color;

uniform vec3 camera_pos;
uniform sampler2D texture_diffuse1;
uniform sampler2D texture_height1;
uniform samplerCube texture_1;

void main() {
  vec3 I = normalize(position - camera_pos);
  vec3 R = reflect(I, normalize(normal));
  vec4 ambient = texture(texture_height1, tex_coord);
  vec4 reflect_color = texture(texture_1, R);
  vec4 origin_color = texture(texture_diffuse1, tex_coord);
  fragment_color = ambient * reflect_color + origin_color;
}