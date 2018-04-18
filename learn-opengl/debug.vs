#version 330 core
layout (location = 0) in vec3 P;
layout (location = 1) in vec2 T;

out vec2 tex_coord;

void main() {
  tex_coord = T;
  gl_Position = vec4(P, 1.0);
}