"""
author: hova88
date: 2021/03/16
"""
import numpy as np 
import numpy as np
from visual_tools import draw_clouds_with_boxes
import open3d as o3d
import yaml

# def cfg_from_yaml_file(cfg_file, config):
#     with open(cfg_file, 'r') as f:
#         try:
#             new_config = yaml.load(f, Loader=yaml.FullLoader)
#         except:
#             new_config = yaml.load(f)
#         merge_new_config(config=config, new_config=new_config)
#     return config

def dataloader(cloud_path , boxes_path):
    ### x, y, z, intensity, time_lag
    cloud = np.loadtxt(cloud_path).reshape(-1, 5)
    #/** use pointpillar box [ x, y, z, dx, dy, dz, yaw, score, cls] */
    boxes = np.loadtxt(boxes_path).reshape(-1, 7) #7
    return cloud , boxes 

if __name__ == "__main__":
    import yaml
    with open("bootstrap.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # input_pcd_file = "/home/hugoliu/github/nvidia/lidar/CenterPoint/tensorrt/data/centerpoint/points/0a0d6b8c2e884134a3b48df43d54c36a.bin"
    # input_pcd_file = "/home/hugoliu/github/nvidia/lidar/PointPillars_MultiHead_40FPS/test/testdata/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801007446751.pcd.txt"
    # output_box_file = "/home/hugoliu/github/nvidia/lidar/CenterPoint/tensorrt/data/centerpoint/fp16_3.txt"
    # output_box_file = "/home/hugoliu/github/nvidia/lidar/PointPillars_MultiHead_40FPS/test/testdata/boxes_n015.txt"
    input_pcd_file = config['InputFile']
    output_box_file = config['OutputFile']
    cloud ,boxes = dataloader(input_pcd_file, output_box_file)
    draw_clouds_with_boxes(cloud ,boxes)