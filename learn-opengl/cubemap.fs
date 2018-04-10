#version 330 core

in vec3 normal;
in vec3 position;

out vec4 fragment_color;

uniform vec3 camera_pos;
uniform samplerCube texture_1;

void main() {
  float ratio = 1.0 / 1.52;
  vec3 I = normalize(position - camera_pos);
  vec3 R = refract(I, normalize(normal), ratio);
  fragment_color = vec4(texture(texture_1, R).rgb, 1.0f);
}