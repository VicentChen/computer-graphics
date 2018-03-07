#version 330 core
in vec4 vertex_color;
in vec2 tex_coord;

out vec4 fragment_color;

uniform sampler2D texture_1;
uniform sampler2D texture_2;
uniform float alpha;

void main() {
  fragment_color = mix(texture(texture_1, tex_coord), texture(texture_2, tex_coord), alpha);
}
