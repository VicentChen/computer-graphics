#version 330 core

in vec3 tex_coord;

out vec4 fragment_color;

uniform samplerCube texture_1;

void main() {
  fragment_color = texture(texture_1, tex_coord);
}