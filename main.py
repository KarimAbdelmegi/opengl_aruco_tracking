import glfw
from OpenGL.GL import *
import ShaderLoader
import numpy as np
from PIL import Image
import pyrr
import math
import cv2



def main():

    def window_resize(window, width, height):
        glViewport(0,0, width, height)
        projection = pyrr.matrix44.create_perspective_projection_matrix(45, width/height, 0.1, 100)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)

    quad_vertices = np.array([
        #positions     texture          normal vectors
        #            coordinates
    
        #top face z-achsis
        -0.0, -0.0, 0.0,  0.0, 0.0,     0.0, 0.0, 1.0,
        0.0, -0.0, 0.0,   1.0, 0.0,     0.0, 0.0, 1.0,
        0.0,  0.0, 0.0,   1.0, 1.0,     0.0, 0.0, 1.0,
        -0.0,  0.0, 0.0,  0.0, 1.0,     0.0, 0.0, 1.0,

        #bottom face z-achsis
        -0.0, -0.0, -0.0,   0.0, 0.0,   0.0, 0.0, -1.0,
        0.0, -0.0, -0.0,    1.0, 0.0,   0.0, 0.0, -1.0,
        0.0,  0.0, -0.0,    1.0, 1.0,   0.0, 0.0, -1.0,
        -0.0,  0.0, -0.0,   0.0, 1.0,   0.0, 0.0, -1.0,

        #right face x-achsis
        0.0, -0.0, -0.0,    0.0, 0.0,   1.0, 0.0, 0.0,
        0.0,  0.0, -0.0,    1.0, 0.0,   1.0, 0.0, 0.0,
        0.0,  0.0,  0.0,    1.0, 1.0,   1.0, 0.0, 0.0,
        0.0, -0.0,  0.0,    0.0, 1.0,   1.0, 0.0, 0.0,

        # left face x-achsis
        -0.0,  0.0, -0.0,   0.0, 0.0,   -1.0, 0.0, 0.0,
        -0.0, -0.0, -0.0,   1.0, 0.0,   -1.0, 0.0, 0.0,
        -0.0, -0.0,  0.0,   1.0, 1.0,   -1.0, 0.0, 0.0,
        -0.0,  0.0,  0.0,   0.0, 1.0,   -1.0, 0.0, 0.0,

        #front face y-achsis
        -0.0, -0.0, -0.0,   0.0, 0.0,   0.0, -1.0, 0.0,
         0.0, -0.0, -0.0,   1.0, 0.0,   0.0, -1.0, 0.0,
        0.0, -0.0,  0.0,    1.0, 1.0,   0.0, -1.0, 0.0,
        -0.0, -0.0,  0.0,   0.0, 1.0,   0.0, -1.0, 0.0,

        #back face y-achsis
         1.0, 1.0, -0.0,    0.0, 0.0,   0.0, 1.0, 0.0,
        -1.0, 1.0, -0.0,    1.0, 0.0,   0.0, 1.0, 0.0,
        -1.0, -1.0,  0.0,    1.0, 1.0,   0.0, 1.0, 0.0,
         1.0, -1.0,  0.0,    0.0, 1.0,   0.0, 1.0, 0.0
    ], dtype=np.float32)


    indices = [0,  1,  2,  2,  3,  0,
            4,  5,  6,  6,  7,  4,
            8,  9, 10, 10, 11,  8,
            12, 13, 14, 14, 15, 12,
            16, 17, 18, 18, 19, 16,
            20, 21, 22, 22, 23, 20]

    indices = np.array(indices, dtype= np.uint32)



    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1) 
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    w_width, w_height = 800, 600

    window = glfw.create_window(w_width, w_height, "OpenGL test", None, None)

    glfw.make_context_current(window)

    glfw.set_window_size_callback(window, window_resize)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    shader = ShaderLoader.compile_shader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl")

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes , quad_vertices, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    #Position attribute
    #position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * quad_vertices.itemsize, ctypes.c_void_p(0))
    

    #Color attribute
    #color = glGetAttribLocation(shader, "color")
    #glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * quad_vertices.itemsize, ctypes.c_void_p(12))
    #glEnableVertexAttribArray(1)

    #texture attributess
    #texture_coords = glGetAttribLocation(shader, "inTexCoords")
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * quad_vertices.itemsize, ctypes.c_void_p(3 * 4))

    #normal vectors attributes
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * quad_vertices.itemsize, ctypes.c_void_p(5 * 4))

    


    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)

    #texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

    #texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    image = Image.open("res/erika_mustermann.png")
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    #img_data = np.array(list(image.getdata()), np.uint8)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)


    glUseProgram(shader)

    glClearColor(0, 0.0, 0.0, 0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, 1280/720, 0.1, 100)
    translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 10]))
    #view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0]))
    
    # first parameter the eye position vector, second parametere the target vector (where the camera is looking at), third parameter is the up vector
    #view = pyrr.matrix44.create_look_at(pyrr.Vector3([2, 1, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    light_loc = glGetUniformLocation(shader, "light")


    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(model_loc, 1, GL_FALSE, translation)

    #glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        #rotation
        camX = math.sin(glfw.get_time()) * 10
        camZ = math.cos(glfw.get_time()) * 10



        # first parameter the eye position vector, second parametere the target vector (where the camera is looking at), third parameter is the up vector
        view = pyrr.matrix44.create_look_at(pyrr.Vector3([camX, 10, camZ]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
        #view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 10]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
        glUniformMatrix4fv(light_loc, 1, GL_FALSE, view)


        #draw the elements
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        # Swap front and back buffers
        glfw.swap_buffers(window)


    glfw.terminate()


if __name__ == "__main__":
    main()