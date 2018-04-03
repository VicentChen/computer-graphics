#version 330 core

in vec2 tex_coords;

out vec4 fragment_color;

uniform sampler2D texture_1;

void main() {
  fragment_color = texture(texture_1, tex_coords);
}