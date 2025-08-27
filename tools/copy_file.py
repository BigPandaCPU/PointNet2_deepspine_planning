import os
import shutil
src_stl_dir = "E:/data/DeepSpineData/label26/bad_stl"
dst_stl_dir = "E:/data/DeepSpineData/label26/bad_landmark_stl"

dst_landmark_dir = "E:/data/DeepSpineData/label26/bad_landmark"

landmark_names = os.listdir(dst_landmark_dir)

for landmark_name in landmark_names:
    src_stl_file = os.path.join(src_stl_dir, landmark_name.replace(".txt", ".stl"))

    if not os.path.exists(src_stl_file):
        continue
    print(landmark_name)
    dst_stl_file = os.path.join(dst_stl_dir, landmark_name.replace(".txt", ".stl"))

    # src_landmark_file = os.path.join(dst_landmark_dir, landmark_name)
    # dst_landmark_file = os.path.join(dst_landmark_dir, landmark_name.replace(".png", "_origin.txt"))

    #shutil.copy(src_landmark_file, dst_landmark_file)
    shutil.copy(src_stl_file, dst_stl_file)

