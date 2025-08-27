import os
import shutil

src_dir = "E:/data/DeepSpineData/label26/bad_landmark_stl"

aim_dir = "E:/data/DeepSpineData/label26/landmark_points"

aim_names = os.listdir(aim_dir)

for aim_name in aim_names:
    src_file = os.path.join(src_dir, aim_name.replace(".txt", ".stl"))
    if os.path.exists(src_file):
        print(src_file)
        os.remove(src_file)
