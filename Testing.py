
import pyrr
import numpy as np


projection_matrix = pyrr.matrix44.create_perspective_projection_matrix(45.0, 1280/720, 0.1, 100.0)



view_matrix = pyrr.matrix44.create_look_at(pyrr.Vector3([0.0, 0.0, 5.0]), pyrr.Vector3([0.0, 0.0, 0.0]), pyrr.Vector3([0.0, 1.0, 0.0]))


combined_matrix = pyrr.matrix44.multiply(projection_matrix, view_matrix)


fov_y = 2.0 * np.arctan(1.0 / combined_matrix[1][1])
aspect_ratio = combined_matrix[1][1] / combined_matrix[0][0]
near_plane = combined_matrix[3][2] / (combined_matrix[2][2] - 1.0)
far_plane = combined_matrix[3][2] / (combined_matrix[2][2] + 1.0)


max_visible_x = near_plane * aspect_ratio * np.tan(fov_y / 2.0)
max_visible_y = near_plane * np.tan(fov_y / 2.0)

print(f"Max Visible X: {max_visible_x}")
print(f"Max Visible Y: {max_visible_y}")