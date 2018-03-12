#define STB_IMAGE_IMPLEMENTATION

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <learnopengl/shader_m.h>
#include <learnopengl/camera.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace glm;

// settings
const int SCR_WIDTH = 800;
const int SCR_HEIGHT = 600;

// camera
Camera camera(vec3(0.0f, 0.0f, 3.0f));
bool first_mouse = true;
float last_X = 0.0f;
float last_Y = 0.0f;

// timing
float delta_time = 0.0f;
float last_frame = 0.0f;
float current_frame = 0.0f;

// lighting
vec3 light_pos(1.2f, 1.0f, 2.0f);

void process_input(GLFWwindow *window);
void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void mouse_callback(GLFWwindow *window, double xpos, double ypos);
void scroll_callback(GLFWwindow *window, double xoffset, double yoffset);
unsigned int load_texture(char *path);

int main(int argc, char* argv[]) {
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Multiple lights", NULL, NULL);
  glfwMakeContextCurrent(window);
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glEnable(GL_DEPTH_TEST);

  // shaders
  Shader lamp_shader("lamp.vs", "lamp.fs");
  Shader cube_shader("multiple-lights.vs", "multiple-lights.fs");

  // vertices
  float vertices[] = {
    // positions          // normals           // texture coords
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
    0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
    0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
    0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
    -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
    0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
    0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
    0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
    -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

    0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
    0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
    0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
    0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
    0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
    0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
    0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
    0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
    -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
    -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
    0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
    -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f
  };

  // positions all containers
  vec3 cube_positions[] = {
    vec3(0.0f,  0.0f,  0.0f),
    vec3(2.0f,  5.0f, -15.0f),
    vec3(-1.5f, -2.2f, -2.5f),
    vec3(-3.8f, -2.0f, -12.3f),
    vec3(2.4f, -0.4f, -3.5f),
    vec3(-1.7f,  3.0f, -7.5f),
    vec3(1.3f, -2.0f, -2.5f),
    vec3(1.5f,  2.0f, -2.5f),
    vec3(1.5f,  0.2f, -1.5f),
    vec3(-1.3f,  1.0f, -1.5f)
  };

  vec3 point_light_positions[] = {
    vec3(0.7f,  0.2f,  2.0f),
    vec3(2.3f, -3.3f, -4.0f),
    vec3(-4.0f,  2.0f, -12.0f),
    vec3(0.0f,  0.0f, -3.0f)
  };

  unsigned lamp_VAO, cube_VAO, VBO;
  glGenVertexArrays(1, &lamp_VAO);
  glGenVertexArrays(1, &cube_VAO);
  glGenBuffers(1, &VBO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);

  glBindVertexArray(lamp_VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  glBindVertexArray(cube_VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  // textures
  unsigned int diffuse_map = load_texture("pic/container2.png");
  unsigned int specular_map = load_texture("pic/container2_specular.png");

  cube_shader.use();
  cube_shader.setInt("material.diffuse", 0);
  cube_shader.setInt("material.specular", 1);

  while (!glfwWindowShouldClose(window)) {
    current_frame = glfwGetTime();
    delta_time = current_frame - last_frame;
    last_frame = current_frame;

    process_input(window);
    
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    mat4 model, view, projection;

    // draw lamp
    for (int i = 0; i < 4; i++) {
      model = mat4();
      model = translate(model, point_light_positions[i]);
      model = scale(model, vec3(0.2f));
      view = camera.GetViewMatrix();
      projection = perspective(radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

      lamp_shader.use();
      lamp_shader.setMat4("model", model);
      lamp_shader.setMat4("view", view);
      lamp_shader.setMat4("projection", projection);

      glBindVertexArray(lamp_VAO);
      glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    // draw cube
    cube_shader.use();
    cube_shader.setVec3("view_pos", camera.Position);
    cube_shader.setFloat("material.shininess", 32.0f);
    // directional light
    cube_shader.setVec3("dir_light.direction", -0.2f, -1.0f, -0.3f);
    cube_shader.setVec3("dir_light.ambient", 0.05f, 0.05f, 0.05f);
    cube_shader.setVec3("dir_light.diffuse", 0.4f, 0.4f, 0.4f);
    cube_shader.setVec3("dir_light.specular", 0.5f, 0.5f, 0.5f);
    // point light 1
    cube_shader.setVec3( "point_light[0].position", point_light_positions[0]);
    cube_shader.setVec3( "point_light[0].ambient", 0.05f, 0.05f, 0.05f);
    cube_shader.setVec3( "point_light[0].diffuse", 0.8f, 0.8f, 0.8f);
    cube_shader.setVec3( "point_light[0].specular", 1.0f, 1.0f, 1.0f);
    cube_shader.setFloat("point_light[0].constant", 1.0f);
    cube_shader.setFloat("point_light[0].linear", 0.09);
    cube_shader.setFloat("point_light[0].quadratic", 0.032);
    // point light 2
    cube_shader.setVec3( "point_light[1].position", point_light_positions[1]);
    cube_shader.setVec3( "point_light[1].ambient", 0.05f, 0.05f, 0.05f);
    cube_shader.setVec3( "point_light[1].diffuse", 0.8f, 0.8f, 0.8f);
    cube_shader.setVec3( "point_light[1].specular", 1.0f, 1.0f, 1.0f);
    cube_shader.setFloat("point_light[1].constant", 1.0f);
    cube_shader.setFloat("point_light[1].linear", 0.09);
    cube_shader.setFloat("point_light[1].quadratic", 0.032);
    // point light 3
    cube_shader.setVec3( "point_light[2].position", point_light_positions[2]);
    cube_shader.setVec3( "point_light[2].ambient", 0.05f, 0.05f, 0.05f);
    cube_shader.setVec3( "point_light[2].diffuse", 0.8f, 0.8f, 0.8f);
    cube_shader.setVec3( "point_light[2].specular", 1.0f, 1.0f, 1.0f);
    cube_shader.setFloat("point_light[2].constant", 1.0f);
    cube_shader.setFloat("point_light[2].linear", 0.09);
    cube_shader.setFloat("point_light[2].quadratic", 0.032);
    // point light 4
    cube_shader.setVec3( "point_light[3].position", point_light_positions[3]);
    cube_shader.setVec3( "point_light[3].ambient", 0.05f, 0.05f, 0.05f);
    cube_shader.setVec3( "point_light[3].diffuse", 0.8f, 0.8f, 0.8f);
    cube_shader.setVec3( "point_light[3].specular", 1.0f, 1.0f, 1.0f);
    cube_shader.setFloat("point_light[3].constant", 1.0f);
    cube_shader.setFloat("point_light[3].linear", 0.09);
    cube_shader.setFloat("point_light[3].quadratic", 0.032);
    // spotLight
    cube_shader.setVec3( "spot_light.position", camera.Position);
    cube_shader.setVec3( "spot_light.direction", camera.Front);
    cube_shader.setVec3( "spot_light.ambient", 0.0f, 0.0f, 0.0f);
    cube_shader.setVec3( "spot_light.diffuse", 1.0f, 1.0f, 1.0f);
    cube_shader.setVec3( "spot_light.specular", 1.0f, 1.0f, 1.0f);
    cube_shader.setFloat("spot_light.constant", 1.0f);
    cube_shader.setFloat("spot_light.linear", 0.09);
    cube_shader.setFloat("spot_light.quadratic", 0.032);
    cube_shader.setFloat("spot_light.cut_off", cos(radians(12.5f)));
    cube_shader.setFloat("spot_light.outer_cut_off", cos(radians(15.0f)));
    for (int i = 0; i < 10; i++) {
      model = mat4();
      model = translate(model, cube_positions[i]);
      model = rotate(model, radians(20.0f * i), vec3(1.0f, 0.3f, 0.5f));
      view = camera.GetViewMatrix();
      projection = perspective(radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);

      cube_shader.setMat4("model", model);
      cube_shader.setMat4("view", view);
      cube_shader.setMat4("projection", projection);

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, diffuse_map);
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, specular_map);
      glBindVertexArray(cube_VAO);
      glDrawArrays(GL_TRIANGLES, 0, 36);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteBuffers(1, &VBO);
  glDeleteVertexArrays(1, &lamp_VAO);
  glDeleteVertexArrays(1, &cube_VAO);

  glfwTerminate();
  return 0;
}

void process_input(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, delta_time);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, delta_time);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, delta_time);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, delta_time);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
  if (first_mouse) {
    last_X = xpos;
    last_Y = ypos;
    first_mouse = false;
  }

  float xoffset = xpos - last_X;
  float yoffset = last_Y - ypos;
  last_X = xpos;
  last_Y = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  camera.ProcessMouseScroll(yoffset);
}

unsigned load_texture(char *path) {
  unsigned int texture;
  int width, height, channels;
  unsigned char *data;
  data = stbi_load(path, &width, &height, &channels, 0);

  GLenum format;
  if (channels == 1)
    format = GL_RED;
  else if (channels == 3)
    format = GL_RGB;
  else if (channels == 4)
    format = GL_RGBA;

  glGenTextures(1, &texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);

  stbi_image_free(data);
  return texture;
}
