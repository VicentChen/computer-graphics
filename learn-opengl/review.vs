#version 330 core
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 color;
layout (location = 2) in vec2 texture_coord;

out vec4 vertex_color;
out vec2 tex_coord;

uniform mat4 transform;

void main() {
  gl_Position = transform * vec4(pos, 1.0f);
  vertex_color = vec4(color, 1.0f);
  tex_coord = texture_coord;
}