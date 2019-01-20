"""
David Hansen Dec. 31, 2018
Basic 3D renderer test with Pygame and OpenGL
"""
# Pygame imports
import pygame
from pygame.locals import *

# OpenGl imports
from OpenGL.GL import *
from OpenGL.GLU import *

# Numpy
import numpy as np

"""Initialization Code"""
# Pygame and OpenGL context initialization
pygame.init()


"""Window code"""
# Display size attributes
display_size = (800, 600)
aspect_ratio = display_size[0] / display_size[1]
pygame.display.set_mode(display_size, DOUBLEBUF | OPENGL)

"""Camera Code"""
# Display camera attributes
fov = 90
near_clip = 0.1
far_clip = 50
cam_pos = (0.0, 0.0, -1.0)
cam_rot = (0.0, 0.0, 0.0, 0.0)

# Sets the OpenGL view context values for basic camera
gluPerspective(fov, aspect_ratio, near_clip, far_clip)

# Sets the initial position and rotation of camera
glTranslatef(cam_pos[0], cam_pos[1], cam_pos[2])
glRotatef(cam_rot[0], cam_rot[1], cam_rot[2], cam_rot[3])

"""Engine Time"""
# Instantiate the clock
time = pygame.time.Clock()


"""Render code"""
# Our model data
model_data = np.array([
    -0.5, -0.5,
    0.0, 0.5,
    0.5, -0.5], dtype=np.float32)

# Generates one buffer and binds that buffer for use
vbo = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vbo)

# Sets what type, how big, and how to optimize model data
glBufferData(GL_ARRAY_BUFFER, len(model_data) * 4, model_data, GL_STATIC_DRAW)

# Defining the data size, type, and how they are indexed
glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 8, None)

# This enables this particular attribute in the vertex array
glEnableVertexAttribArray(0)


# This function compiles a shader program
def compile_shader(type, source):
    """
    Compiles shaders
    :param type: A GL_XXXXXXX shader type
    :param source: The shader code string
    :return: shader program ID
    """

    # Generates an id for the shader
    id = glCreateShader(type)

    # Sends the source to be compiled to OpenGL and what ID it will have
    glShaderSource(id, source)

    # Compiles shade source
    glCompileShader(id)

    # Error checking code
    if glGetShaderiv(id, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(id))

    return id


# This function is to simplify shader creation
def create_shader(vertex_shader, fragment_shader):
    """
    Creates shaders by setting context variables. Returns a program ID  pointing to the shaders made.
    :param vertex_shader: Vertex shade code
    :param fragment_shader: Fragment shader code
    :return: Shader program ID
    """

    # Creates a program, much like genbuffers/createshader
    program = glCreateProgram()

    # Calls our function and compiles shaders
    vs = compile_shader(GL_VERTEX_SHADER, vertex_shader)
    fs = compile_shader(GL_FRAGMENT_SHADER, fragment_shader)

    # Attaches the shaders to te program ID
    glAttachShader(program, vs)
    glAttachShader(program, fs)

    # Links them to the program
    glLinkProgram(program)

    # Error checking for linking
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))

    # More checking
    glValidateProgram(program)

    # Deletes the intermediate objects made. This  isn't the shader, just intermediate stuff
    glDeleteShader(vs)
    glDeleteShader(fs)

    return program


# Vertex shader code
vertex_shader = """
#version 330 core

layout(location = 0) in vec4 position;

void main()
{
    gl_Position = position;
}
"""

# Fragment shader code
fragment_shader = """
#version 330 core

layout(location = 0) out vec4 color;

void main()
{
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

# Make shaders in the context then return the program ID
shader = create_shader(vertex_shader, fragment_shader)

# Use the  shaders we just made and use the program id returned by var shader
glUseProgram(shader)

"""Start of main"""
# The BEGINNING!!!
if __name__ == "__main__":

    # Main game loop
    while True:

        # Event logic
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Clears the screen buffers to prepare for next frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # TODO add more render code here
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # Flips the buffers to present new image
        pygame.display.flip()

        # Pumps event messages from event queue
        pygame.event.pump()

        # The frames per second
        time.tick(100)

# Cleans up the shaders
glDeleteProgram(shader)
