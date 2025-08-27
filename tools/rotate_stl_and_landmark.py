import os
import numpy as np
import random
from vtk_tools import *

if __name__ == "__main__":
    axis_dir = "E:/data/stl/good_landmark_points/axis"
    #stl_dir = "E:/data/stl/good_landmark_points/stl_test"
    src_stl_dir = "E:/data/stl/good_landmark_points/stl_train"
    src_landmark_dir = "E:/data/stl/good_landmark_points/landmark_train"

    dst_stl_dir = "E:/data/stl/good_landmark_points/stl_train_ext"
    dst_landmark_dir = "E:/data/stl/good_landmark_points/landmark_train_ext"

    os.makedirs(dst_landmark_dir, exist_ok=True)
    os.makedirs(dst_stl_dir, exist_ok=True)






    axis_names = os.listdir(axis_dir)
    for axis_name in axis_names:
        # if "new_Spine" not in axis_name:
        #     continue

        landmark_file = os.path.join(src_landmark_dir, axis_name)

        if not os.path.exists(landmark_file):
            continue
        landmark_points = getLandmarkPoints(landmark_file)

        stl_file = os.path.join(src_stl_dir, axis_name.replace(".txt", ".stl"))
        if not os.path.exists(stl_file):
            continue


        print(axis_name)

        target_poly_data = createPolyDataFromSTL(stl_file)

        #target_actor = createActorFromPolydata(target_poly_data, color='yellow')

        axis_file = os.path.join(axis_dir, axis_name)
        axis_info = getAxisInfo(axis_file)

        spine_axis_center = axis_info['axis_center']
        spine_axis_normalX = axis_info['axis_normalX']
        spine_axis_normalY = axis_info['axis_normalY']
        spine_axis_normalZ = axis_info['axis_normalZ']
        NUM_COUNT = 5
        for i in range(NUM_COUNT):

            matrix1 = np.eye(4)
            matrix1[0:3, 3] = -spine_axis_center


            rotZ = random_integer = random.randint(0, 180)
            matrix2 = np.eye(4)
            rotate_matrix = createRotateMatrixAroundNormal(spine_axis_normalZ, rotZ)
            matrix2[0:3, 0:3] = rotate_matrix

            rotY = random_integer = random.randint(0, 180)
            matrix3 = np.eye(4)
            rotate_matrix = createRotateMatrixAroundNormal(spine_axis_normalY, rotY)
            matrix3[0:3, 0:3] = rotate_matrix

            rotX = random_integer = random.randint(0, 180)
            matrix4 = np.eye(4)
            rotate_matrix = createRotateMatrixAroundNormal(spine_axis_normalX, rotX)
            matrix4[0:3, 0:3] = rotate_matrix

            matrix5 = np.eye(4)
            matrix5[0:3, 3] = spine_axis_center



            # matrix_np = np.dot(matrix5, np.dot(matrix4, np.dot(matrix3, np.dot(matrix2, matrix1))))
            matrix_np = np.dot(np.dot(np.dot(matrix4, matrix3), matrix2), matrix1)
            # 这里构造左乘的矩阵
            matrix_vtk = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    matrix_vtk.SetElement(i, j, matrix_np[i, j])


            trans = vtk.vtkTransform()
            trans.SetMatrix(matrix_vtk)

            transformFilter = vtk.vtkTransformPolyDataFilter()
            transformFilter.SetInputData(target_poly_data)
            transformFilter.SetTransform(trans)
            transformFilter.Update()

            save_stl_file = os.path.join(dst_stl_dir, axis_name.replace(".txt", "_rotZ%d_rotY%d_rotX%d.stl"%(rotZ,rotY,rotX)))
            stl_writer = vtk.vtkSTLWriter()
            stl_writer.SetFileName(save_stl_file)  # 输出 STL 文件名
            stl_writer.SetInputData(transformFilter.GetOutput())  # 设置输入数据
            stl_writer.SetFileTypeToBinary()  # 设置文件类型为二进制
            stl_writer.Write()

            all_actors = []


            save_landmark_file = os.path.join(dst_landmark_dir, axis_name.replace(".txt", "_rotZ%d_rotY%d_rotX%d.txt"%(rotZ,rotY,rotX)))

            fp = open(save_landmark_file, 'w')
            for k, points in landmark_points.items():
                if k == "0":
                    color = "red"
                if k == "1":
                    color = "green"
                if k == "2":
                    color = "blue"
                points_extend = np.ones([len(points), 4])
                points_extend[:,0:3] = points
                #points = np.dot(points - spine_axis_center, rotate_matrix[0:3, 0:3].T)+spine_axis_center
                points_extend_new = np.dot(points_extend, matrix_np.T)
                points = points_extend_new[:, 0:3]
                # points_actors = createPointsActor(points, radius=1.0, opacity=1.0, color=color)
                # all_actors.extend(points_actors)
                for point in points:
                    fp.write("%s,%f,%f,%f\n"%(k, point[0], point[1], point[2]))
            fp.close()

            # trans_actor = createActorFromPolydata(transformFilter.GetOutput())
            # #
            # all_actors.append(trans_actor)
            # #all_actors.append(target_actor)
            # showActors(all_actors)
        #break


