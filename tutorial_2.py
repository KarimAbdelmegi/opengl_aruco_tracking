from OpenGL.GL import *
import glfw
import ShaderLoader
import numpy as np
from PIL import Image
import pyrr
import glm
import cv2
from TextureLoader import load_texture
import time
import math


def main():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_param = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)

    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # charuco_board = cv2.aruco.CharucoBoard((10, 7), 15, 11, aruco_dict)
    # charuco_param = cv2.aruco.CharucoParameters()
    # detector = cv2.aruco.CharucoDetector(charuco_board, charuco_param,)

    cameraMatrix = np.load("CameraMatrix.npy")
    distortionCoef = np.load("Distortion.npy")
    real_size_aruco_marker = 100

    card_aspect_r = 1.586


    cap = cv2.VideoCapture(0)
    w_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    aspect_ratio = w_width/w_height

    card_z_pos =  -0.7
    card_x_pos =  aspect_ratio * 0
    card_y_pos = 0

    fov_y = 45
    near_plane = 0.1
    far_plane = 100

    prev_rvec = None

    background_z_pos = -50
    

    def window_resize(window, width, height):
        glViewport(0,0, width, height)
        projection = pyrr.matrix44.create_perspective_projection_matrix(fov_y, aspect_ratio, near_plane, far_plane)
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)


    def update_tex_frame(texture):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)
        if not ret:
            return
        
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.shape[1], frame.shape[0], 0, GL_BGR, GL_UNSIGNED_BYTE, frame)

        return texture, frame

    def visible_region_coordinates(z_depth, view_matrix, projection_matrix):
        # Calculate inverse matrices
        inv_projection = np.linalg.inv(projection_matrix)
        inv_view = np.linalg.inv(view_matrix)

        # Calculate near and far planes intersection points
        near_plane_point = inv_projection @ pyrr.Vector4([0, 0, -1, 1])
        near_plane_point /= near_plane_point.w  # Homogeneous divide
        near_plane_point = inv_view @ near_plane_point

        far_plane_point = inv_projection @ pyrr.Vector4([0, 0, 1, 1])
        far_plane_point /= far_plane_point.w
        far_plane_point = inv_view @ far_plane_point

        # Interpolate coordinates at the specified depth
        interpolation_factor = (z_depth - near_plane_point.z) / (far_plane_point.z - near_plane_point.z)
        visible_coordinates = near_plane_point + (far_plane_point - near_plane_point) * interpolation_factor

        return visible_coordinates


    cube_vertices = np.array([
        #positions     texture          normal vectors
        #            coordinates
    
        #top face z-achsis
        -1.0 * card_aspect_r, -1, -0.0,  0.0, 0.0,    0.0, 0.0, 1.0,
        1.0 * card_aspect_r, -1, 0.0,   1.0, 0.0,     0.0, 0.0, 1.0,
        1.0 * card_aspect_r,  1, 0.0,   1.0, 1.0,     0.0, 0.0, 1.0,
        -1.0 * card_aspect_r,  1, 0.0,  0.0, 1.0,     0.0, 0.0, 1.0
    ], dtype=np.float32)


    cube_indices = [0,  1,  2,  2,  3,  0]

    cube_indices = np.array(cube_indices, dtype= np.uint32)


    quad_vertices = np.array([
                -1 * aspect_ratio, -1, 0,      0.0, 0.0,
                  1 * aspect_ratio, -1, 0,     1.0, 0.0,
                  1 * aspect_ratio,  1, 0,     1.0, 1.0,
                 -1 * aspect_ratio,  1, 0,     0.0, 1.0
                 ], dtype=np.float32)
    

    
    quad_indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)


    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1) 
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)



    window = glfw.create_window(int(w_width), int(w_height), "OpenGL test", None, None)

    glfw.make_context_current(window)

    glfw.set_window_size_callback(window, window_resize)

    cube_VAO = glGenVertexArrays(1)
    glBindVertexArray(cube_VAO)

    cube_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, cube_VBO)
    glBufferData(GL_ARRAY_BUFFER, cube_vertices.nbytes , cube_vertices, GL_STATIC_DRAW)

    cube_EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cube_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, cube_indices.nbytes, cube_indices, GL_STATIC_DRAW)

    #Position attribute
    #position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * cube_vertices.itemsize, ctypes.c_void_p(0))

    #Color attribute
    #color = glGetAttribLocation(shader, "color")
    #glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * cube_vertices.itemsize, ctypes.c_void_p(12))
    #glEnableVertexAttribArray(1)

    #texture attributess
    #texture_coords = glGetAttribLocation(shader, "inTexCoords")
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 8 * cube_vertices.itemsize, ctypes.c_void_p(3 * 4))

    #normal vectors attributes
    glEnableVertexAttribArray(2)
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 8 * cube_vertices.itemsize, ctypes.c_void_p(5 * 4))


    quad_VAO = glGenVertexArrays(1)
    glBindVertexArray(quad_VAO)

    quad_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, quad_VBO)
    glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes , quad_vertices, GL_STATIC_DRAW)

    quad_EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, quad_EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, quad_indices.nbytes, quad_indices, GL_STATIC_DRAW)

    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(0))

    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, quad_vertices.itemsize * 5, ctypes.c_void_p(12))
    

    texture = glGenTextures(2)
    card_texture = load_texture(texture[0], path ="res/erika_mustermann.png", flip=True)
    #background_texture = load_texture(texture[1], path ="res/face_eagle.png")
    

    shader = ShaderLoader.compile_shader("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl")

    glUseProgram(shader)

    glClearColor(0, 0.0, 0.0, 0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    projection = pyrr.matrix44.create_perspective_projection_matrix(fov_y, aspect_ratio, near_plane, far_plane)
    #projection = pyrr.matrix44.create_perspective_projection_matrix_from_bounds(left=-aspect_ratio, right=aspect_ratio, bottom=-1, top=1, near=0.1, far=100.0)

    #view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))
    view = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, -10]))
    
    # first parameter the eye position vector, second parametere the target vector (where the camera is looking at), third parameter is the up vector
    #view = pyrr.matrix44.create_look_at(pyrr.Vector3([2, 1, 3]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")
    light_loc = glGetUniformLocation(shader, "light")
    transform_loc = glGetUniformLocation(shader, "transform")


    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    
    visible = visible_region_coordinates(-2, view, projection)

    while not glfw.window_should_close(window):
        glfw.poll_events()

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        background_texture, frame = update_tex_frame(texture[1])
        frame = cv2.flip(frame, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_corners, aruco_ids, reject = detector.detectMarkers(gray)
        #charuco_corners, charuco_ids, rejected_corners, rejected_ids = detector.detectBoard(gray)


        if np.shape(aruco_corners) == (1, 1, 4, 2):
            for ids, corners in zip(aruco_ids, aruco_corners):
                corners = np.array(corners).astype(int)
                cv2.polylines(frame, [corners], True, (0, 255, 255), thickness=10)
            aruco_corners = np.reshape(aruco_corners, (4,2))
            x1 = aruco_corners[0][0]
            x2 = aruco_corners[1][0]
            

            if( x2 > x1 ) : width_of_marker = x2 - x1
            else : width_of_marker = x1 - x2

            size_of_marker_pixels = width_of_marker 
            size_of_marker_scaled = size_of_marker_pixels / real_size_aruco_marker

            object_points = np.array([[0, 0, 0],
                                      [real_size_aruco_marker, 0, 0],
                                      [real_size_aruco_marker, real_size_aruco_marker, 0],
                                      [0, real_size_aruco_marker, 0]], dtype=np.float32)
            
            image_points = np.array(aruco_corners, dtype=np.float32)
            #object_points = charuco_board.getChessboardCorners()
            #charuco_corners = np.reshape(charuco_corners, (len(charuco_corners), 2))

            #if len(object_points) == len(charuco_corners):
            _, rvec, tvec = cv2.solvePnP(object_points, image_points, cameraMatrix, distortionCoef)

            rvec = rvec.flatten()
            tvec = tvec.flatten()

            rvec[0] = -rvec[0]

            #print("X: ",tvec[0])
            #print("Y: ",tvec[1])
            #print("Z: " ,tvec[2])

            if prev_rvec is not None:
                alpha = 0.9
                rvec = (alpha * rvec + (1 - alpha) * prev_rvec)


            cv2.drawFrameAxes(frame, cameraMatrix, distortionCoef, rvec, tvec, 100)

            #tvec = tvec.flatten() / (size_of_marker_scaled * 100)
            rotation_matrix_3x3 = np.array(glm.mat3(cv2.Rodrigues(rvec)[0]))
            #print("Rotation matrix: \n", rotation_matrix_3x3)

            rotation_matrix_3x3[1:3, :] = -rotation_matrix_3x3[1:3, :]

            rotation_matrix = np.array(glm.mat4(1.0))
            rotation_matrix[:3, :3] = rotation_matrix_3x3

            prev_rvec = rvec
            

        cv2.imshow("frame", frame)

        glUniformMatrix4fv(light_loc, 1, GL_FALSE, view)

        try:
            #adjusted_x_card = 10 * card_z_pos * (w_width/2 - aruco_corners[0][0]) / (w_width/2)
            #adjusted_y_card = -10 * card_z_pos * (w_height/2 - aruco_corners[0][1]) / (w_height/2)

            tvec_normalized = tvec/np.linalg.norm(tvec)

            adjusted_z_card = -((tvec[2]-350)/34)
            adjusted_x_card = 1.7 * tvec_normalized[0] *  (7.35 + (-adjusted_z_card * 0.74))
            adjusted_y_card = -2.9 * tvec_normalized[1] *  (4.14 + (-adjusted_z_card * 0.41))
            
            #print("X_norm: ", tvec_normalized[0])
            print("Y_norm: ", tvec_normalized[1])
            #print("Y: ", tvec[1])
            #print("Z: ", tvec[2])

            #print("Adjusted X: ", adjusted_x_card)
            print("Adjusted Y: ", adjusted_y_card)
            print("Adjusted Z: ", adjusted_z_card)

            

            #print(tvec_normalized)

            cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([adjusted_x_card, adjusted_y_card, adjusted_z_card]))
            #cube_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3([adjusted_x_card, adjusted_y_card, -1]))
            rot_x = pyrr.Matrix44.from_x_rotation(0.5 * glfw.get_time())
            rot_y = pyrr.Matrix44.from_y_rotation(0.8 * glfw.get_time())

            rotation = pyrr.matrix44.multiply(rot_x, rot_y)
            cube_model2 = pyrr.matrix44.multiply(rotation_matrix, cube_pos)
            cube_model3 = pyrr.matrix44.multiply(rotation, cube_pos)

            glBindVertexArray(cube_VAO)
            glBindTexture(GL_TEXTURE_2D, card_texture)
            glUniformMatrix4fv(model_loc, 1, GL_FALSE, cube_model2)
            #glUniformMatrix4fv(transform_loc, 1, GL_FALSE, cube_model2)
            glUniformMatrix4fv(light_loc, 1, GL_FALSE, cube_model2)
            glDrawElements(GL_TRIANGLES, len(cube_indices), GL_UNSIGNED_INT, None)

        except:
            pass

        quad_pos2 = glm.vec3(0.0, 0.0, background_z_pos)
        quad_model = glm.mat4(1.0)
        quad_model = glm.translate(quad_model, quad_pos2)
        quad_model = glm.scale(quad_model, glm.vec3(background_z_pos * (-0.5), background_z_pos * (-0.5), 0))


        quad_model = np.array(quad_model)
        quad_model = quad_model.transpose()
        glBindVertexArray(quad_VAO)
        glBindTexture(GL_TEXTURE_2D, background_texture)
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, quad_model)
        #glUniformMatrix4fv(transform_loc, 1, GL_FALSE, quad_model)
        glUniformMatrix4fv(light_loc, 1, GL_FALSE, quad_model)
        glDrawElements(GL_TRIANGLES, len(quad_indices), GL_UNSIGNED_INT, None)

        # Swap front and back buffers
        glfw.swap_buffers(window)
        #time.sleep(0.5)



    glfw.terminate()


if __name__ == "__main__":
    main()