#version 330 core

layout (location = 0) in vec3 pos;

out vec3 tex_coord;

uniform mat4 view; 
uniform mat4 projection;

void main() {
  vec4 temp = projection * view * vec4(pos, 1.0f);
  tex_coord = pos;
  gl_Position = temp.xyww;
}