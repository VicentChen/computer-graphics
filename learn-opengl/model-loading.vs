#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 norm;
layout (location = 2) in vec2 texture_coordinate;

out vec2 tex_coord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
	tex_coord = texture_coordinate;
	gl_Position = projection * view * model * vec4(pos, 1.0f);
}