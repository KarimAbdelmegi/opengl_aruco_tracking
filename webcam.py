import glfw
from OpenGL.GL import *
import ShaderLoader
import numpy as np
from PIL import Image
import pyrr
import math
import cv2



def main():

    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1) 
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    w_width, w_height = 800, 600

    window = glfw.create_window(w_width, w_height, "OpenGL test", None, None)

    glfw.make_context_current(window)


    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    


    texture_id1 = glGenTextures(1)
    texture_id2 = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id1)

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


    cap = cv2.VideoCapture(0)

    glEnable(GL_TEXTURE_2D)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glDepthFunc(GL_LESS)
    glEnable(GL_DEPTH_TEST)

    texture_id1 = glGenTextures(1)

    while not glfw.window_should_close(window):

        ret, frame = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        glBindTexture(GL_TEXTURE_2D, texture_id1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)

        # Clear the screen and set up the view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw the texture on a quad
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, 1.0, 0.0)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, 1.0, 0.0)
        glEnd()

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # Poll for and process events
        glfw.poll_events()


    glfw.terminate()
    cap.release()


if __name__ == "__main__":
    main()