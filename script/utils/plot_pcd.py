import os
import open3d
import numpy as np

pcd_file = r'/Users/williamed/Desktop/WorkSpace/temp/frame-000000.bin'
pcd_set = np.fromfile(pcd_file, dtype=np.float32).reshape(-1, 3)

########## Method1: PointCloud Visualization for 7Scenes ############################
# pc = open3d.geometry.PointCloud()
# pc.points = open3d.utility.Vector3dVector(pcd_set)
# ds_pcd = pc.voxel_down_sample(voxel_size=0.01)
# open3d.visualization.draw_geometries([ds_pcd])

########## Method2: PointCloud Visualization for 7Scenes ############################
vis = open3d.visualization.Visualizer()
vis.create_window(window_name=os.path.basename(__file__))
render_option = vis.get_render_option()
render_option.background_color = np.array([0.1529, 0.1569, 0.1333], np.float32)
render_option.point_color_option = open3d.visualization.PointColorOption.ZCoordinate
render_option.point_size = 0.5
pcd = open3d.geometry.PointCloud()
pcd.points = open3d.utility.Vector3dVector(pcd_set)
down_sample_pcd = pcd.voxel_down_sample(voxel_size=0.01)
downl_submap_pcd_points = np.array(down_sample_pcd.points)
print(len(downl_submap_pcd_points))
vis.add_geometry(down_sample_pcd)
view_control = vis.get_view_control()
params = view_control.convert_to_pinhole_camera_parameters()
view_control.convert_from_pinhole_camera_parameters(params)
vis.run()