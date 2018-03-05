#include <glad/glad.h>
#include <GLFW/glfw3.h>

const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

const char *vertex_shader_source = ""
"#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"void main() {\n"
"  gl_Position = vec4(aPos, 1.0f); // set position\n"
"}\0";

const char *fragment_shader_source_orange = ""
"#version 330 core\n"
"out vec4 FragColor;\n"
"void main() {\n"
"  FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); // set color to orange\n"
"}\0";

const char *fragment_shader_source_yellow = ""
"#version 330 core\n"
"out vec4 FragColor;\n"
"void main() {\n"
"  FragColor = vec4(1.0f, 1.0f, 0.0f, 1.0f); // set color to yellow\n"
"}\0";

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void process_input(GLFWwindow *window);

int main(int argc, char* argv[]) {

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Hello Triangle", NULL, NULL);
  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // build and compile our shader program
  // ------------------------------------

  int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader);

  int fragment_shader_orange = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader_orange, 1, &fragment_shader_source_orange, NULL);
  glCompileShader(fragment_shader_orange);

  int fragment_shader_yellow = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader_yellow, 1, &fragment_shader_source_yellow, NULL);
  glCompileShader(fragment_shader_yellow);

  int orange_shader_program = glCreateProgram();
  glAttachShader(orange_shader_program, vertex_shader);
  glAttachShader(orange_shader_program, fragment_shader_orange);
  glLinkProgram(orange_shader_program);

  int yellow_shader_program = glCreateProgram();
  glAttachShader(yellow_shader_program, vertex_shader);
  glAttachShader(yellow_shader_program, fragment_shader_yellow);
  glLinkProgram(yellow_shader_program);

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader_orange);
  glDeleteShader(fragment_shader_yellow);

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float vertices_1[] = {
    // first triangle
    0.5f,  0.5f, 0.0f,  // top right
    0.5f, -0.5f, 0.0f,  // bottom right
    -0.5f,  0.5f, 0.0f,  // top left 
  };

  float vertices_2[] = {
    // second triangle
    0.5f, -0.5f, 0.0f,  // bottom right
    -0.5f, -0.5f, 0.0f,  // bottom left
    -0.5f,  0.5f, 0.0f   // top left
  };

  unsigned int VAO_1, VBO_1, VAO_2, VBO_2;
  glGenVertexArrays(1, &VAO_1);
  glGenBuffers(1, &VBO_1);
  
  glBindVertexArray(VAO_1);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_1);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_1), vertices_1, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glGenVertexArrays(1, &VAO_2);
  glGenBuffers(1, &VBO_2);

  glBindVertexArray(VAO_2);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_2);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices_2), vertices_2, GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  while (!glfwWindowShouldClose(window)) {
    process_input(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(orange_shader_program);
    glBindVertexArray(VAO_1);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glUseProgram(yellow_shader_program);
    glBindVertexArray(VAO_2);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height) { 
  glViewport(0, 0, width, height);
}

void process_input(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}
