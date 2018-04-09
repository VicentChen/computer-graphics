#version 330 core

layout (location = 0) in vec2 pos;
layout (location = 1) in vec2 texture_coordinate;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec2 tex_coords;

void main() {
  gl_Position = vec4(pos.x, pos.y, 0.0f, 1.0f);
  tex_coords = texture_coordinate;
}