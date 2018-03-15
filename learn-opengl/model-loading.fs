#version 330 core
in vec2 tex_coord;

out vec4 fragment_color;

uniform sampler2D texture_diffuse1;

void main() {
	fragment_color = texture(texture_diffuse1, tex_coord);
}