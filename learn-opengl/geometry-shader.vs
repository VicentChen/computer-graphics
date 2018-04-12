#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 2) in vec2 texture_coordinate;

out VS_OUT {
	vec2 tex_coord;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	vs_out.tex_coord = texture_coordinate;
	gl_Position = projection * view * model * vec4(pos, 1.0f);
}