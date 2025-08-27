import os
import numpy as np
from vtk_tools import createPointsActor
from vtk_tools import createActorFromSTL
from vtk_tools import showActors

def getLandmarkPoints(landmark_file):
    """

    """
    fp = open(landmark_file)
    lines = fp.readlines()
    fp.close()
    landmark_points = {}
    for line in lines:
        line_list = line.strip().split(",")
        k = line_list[0]
        x = float(line_list[1])
        y = float(line_list[2])
        z = float(line_list[3])
        if k not in landmark_points.keys():
            landmark_points[k] = []
        landmark_points[k].append([x,y,z])
    return landmark_points


if __name__ == "__main__":
    landmark_dir = "D:/project/PyProject/PointNet2_deepspine/data/landmarks"
    stl_dir = "D:/project/PyProject/PointNet2_deepspine/data/registration_test_data"

    # landmark_names = []
    # sub_dirs = [sub_dir for sub_dir in os.listdir(landmark_dir) if os.path.isdir(os.path.join(landmark_dir, sub_dir))]
    # for sub_dir in sub_dirs:
    #     sub_path = os.path.join(landmark_dir, sub_dir)
    #     landmark_names.extend(os.listdir(sub_path))

    landmark_names = os.listdir(landmark_dir) #[name for name in os.listdir(landmark_dir) if os.path.isfile(os.path.join(landmark_dir, name))]

    for landmark_name in landmark_names:
        if ".txt" not in landmark_name:
            continue
        # if "origin_rotz.txt" not in landmark_name:
        #     continue
        # if "Test86_1" not in landmark_name:
        #     continue
        # if "_23.txt" not in landmark_name:
        #     continue
        print(landmark_name)
        landmark_file = os.path.join(landmark_dir, landmark_name)
        stl_file = os.path.join(stl_dir, landmark_name.replace(".txt", ".stl"))
        if not os.path.exists(stl_file):
            stl_file = os.path.join(stl_dir, "tmp", landmark_name.replace(".txt", ".stl"))

        all_actors = []
        target_actor = createActorFromSTL(stl_file)
        all_actors.append(target_actor)
        landmark_points = getLandmarkPoints(landmark_file)

        for label, points in landmark_points.items():
            if label == "0":
                color="red"
            if label == "1":
                color = "green"
            if label == "2":
                color = "blue"
            cur_points_actors = createPointsActor(points, radius=0.5, opacity=1.0, color=color)
            all_actors.extend(cur_points_actors)
        showActors(all_actors)


