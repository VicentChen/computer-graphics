#version 330 core

in vec2 tex_coords;

out vec4 fragment_color;

uniform sampler2D texture_1;

void main() {
  vec4 tex_color = texture(texture_1, tex_coords);
  if (tex_color.a < 0.1f) discard;
  fragment_color = tex_color;
}