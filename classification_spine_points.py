import os
import numpy as np
import SimpleITK as sitk
from vtk_tools import *
import vtk

import torch
from torch.autograd import Variable
import pointnet2_part_seg_msg as MODEL

def saveSpineSurgicalPlanning2Png(all_actors, axis_center, axis_normalZ, axis_normalY, axis_normalX, save_png_file):
    """
    """
    ren = vtk.vtkRenderer()
    for cur_actor in all_actors:
        ren.AddActor(cur_actor)
    #ren.SetAntialiasing(True)

    win = vtk.vtkRenderWindow()
    win.AddRenderer(ren)
    win.SetWindowName("show spine")
    win.SetSize(1000, 1000)

    win.SetMultiSamples(4)

    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(win)
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)


    a_camera = vtkCamera()
    camera_pos1 = axis_center + 10.0 * axis_normalZ
    a_camera.SetPosition(camera_pos1[0], camera_pos1[1], camera_pos1[2])
    a_camera.SetViewUp(-axis_normalY[0], -axis_normalY[1], -axis_normalY[2])
    a_camera.SetFocalPoint(axis_center[0], axis_center[1], axis_center[2])
    a_camera.ComputeViewPlaneNormal()
    a_camera.Zoom(2.0)
    # a_camera.Azimuth(180.0)
    # a_camera.Elevation(10.0)
    ren.SetActiveCamera(a_camera)
    ren.ResetCamera()
    # a_camera.Zoom(1.3)
    # ren.SetActiveCamera(a_camera)
    win.ShowWindowOff()
    win.Render()

    w2if1 = vtk.vtkWindowToImageFilter()
    w2if1.SetInput(win)
    w2if1.SetInputBufferTypeToRGB()
    w2if1.ReadFrontBufferOff()
    w2if1.Update()
    #pic1 = w2if1.GetOutput()

    camera_pos2 = axis_center - 5.0*axis_normalZ + 10.0 * axis_normalX
    a_camera.SetPosition(camera_pos2[0], camera_pos2[1], camera_pos2[2])
    a_camera.SetFocalPoint((axis_center - 5.0*axis_normalZ)[0], (axis_center - 5.0*axis_normalZ)[1], (axis_center - 5.0*axis_normalZ)[2])
    ren.SetActiveCamera(a_camera)
    ren.ResetCamera()
    # a_camera.Zoom(1.3)
    # ren.SetActiveCamera(a_camera)
    win.Render()
    w2if2 = vtk.vtkWindowToImageFilter()
    w2if2.SetInput(win)
    w2if2.SetInputBufferTypeToRGB()
    w2if2.ReadFrontBufferOff()
    w2if2.Update()

    # camera_pos3 = points_center + 40.0 * points_y_vector
    # a_camera.SetPosition(camera_pos3[0], camera_pos3[1], camera_pos3[2])
    # ren.SetActiveCamera(a_camera)
    # ren.ResetCamera()
    # win.Render()
    # # a_camera.Zoom(1.3)
    # # ren.SetActiveCamera(a_camera)
    # w2if3 = vtkWindowToImageFilter()
    # w2if3.SetInput(win)
    # w2if3.SetInputBufferTypeToRGB()
    # w2if3.ReadFrontBufferOff()
    # w2if3.Update()

    # camera_pos4 = points_center - 40.0 * points_y_vector
    # a_camera.SetPosition(camera_pos4[0], camera_pos4[1], camera_pos4[2])
    # ren.SetActiveCamera(a_camera)
    # ren.ResetCamera()
    # win.Render()
    # # a_camera.Zoom(1.3)
    # # ren.SetActiveCamera(a_camera)
    # w2if4 = vtk.vtkWindowToImageFilter()
    # w2if4.SetInput(win)
    # w2if4.SetInputBufferTypeToRGB()
    # w2if4.ReadFrontBufferOff()
    # w2if4.Update()

    imageAppend = vtk.vtkImageAppend()
    imageAppend.SetInputConnection(w2if1.GetOutputPort())
    imageAppend.AddInputConnection(w2if2.GetOutputPort())
    # imageAppend.AddInputConnection(w2if3.GetOutputPort())
    # imageAppend.AddInputConnection(w2if4.GetOutputPort())

    #
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(save_png_file)
    writer.SetInputConnection(imageAppend.GetOutputPort())
    writer.Write()

def parsePointsFile(points_file):
    """
    func:get points from point files
    :param points_file:
    :return:
    """
    fp = open(points_file, "r")
    lines = fp.readlines()
    fp.close()
    cur_points_dict = {}
    for cur_line in lines:
        cur_line = cur_line.strip()
        if "<point" in cur_line:
            cur_line = cur_line.replace("/>", "")
            cur_line_lists = cur_line.split(" ")
            cur_point_x = 0
            cur_point_y = 0
            cur_point_z = 0
            cur_point_name = ""
            for cur_line_list in cur_line_lists:
                if "x=" in cur_line_list:
                    cur_point_x = float(cur_line_list[3:-1])
                if "y=" in cur_line_list:
                    cur_point_y = float(cur_line_list[3:-1])
                if "z=" in cur_line_list:
                    cur_point_z = float(cur_line_list[3:-1])
                if "name=" in cur_line_list:
                    cur_point_name = cur_line_list[6:-1]
            #print(("%s:x=%.2f y=%.2f z=%.2f")%(cur_point_name, cur_point_x, cur_point_y, cur_point_z))
            cur_points_dict[cur_point_name] = np.array([cur_point_x, cur_point_y, cur_point_z])
    return cur_points_dict

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def predict(points, model_dir='./log/part_seg/2023-03-31_12-56'):
    """
    :param points:
    :return:
    """

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    experiment_dir = model_dir

    num_classes = 1
    num_part = 4
    normal = False

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    classifier = MODEL.get_model(num_part, normal_channel=normal).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    point_set = pc_normalize(points)
    point_set = torch.from_numpy(point_set)

    # batchsize, num_point, _ = point_set.size()
    # cur_batch_size, NUM_POINT, _ = point_set.size()
    points = point_set.float().cuda()
    points = Variable(points.view(1, points.size()[0], points.size()[1]))
    points = points.transpose(2, 1)

    # vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
    label = np.array([0])
    label = torch.from_numpy(label)
    label = label.long().cuda()
    seg_pred, _ = classifier(points, to_categorical(label, num_classes))
    seg_pred = seg_pred.cpu().detach().numpy()
    seg_pred = np.argmax(seg_pred, axis=2)
    seg_pred = seg_pred[0]
    return seg_pred


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def PedicleSurgeryPlanning(landmark_points_t, landmark_points_l, landmark_points_r, target_poly_data, TwoCentersDistanceThreshold=3, window_name="show spine"):
    """

    """
    all_actors = []

    left_cut_plane_center = None
    left_cut_plane_normal = None
    right_cut_plane_center = None
    right_cut_plane_normal = None

    top_points_actors = createPointsActor(landmark_points_t, radius=0.5, opacity=1.0, color='red')
    all_actors.extend(top_points_actors)

    target_actor = createActorFromPolydata(target_poly_data, opacity=0.8)
    all_actors.append(target_actor)

    landmark_points_l_center = np.mean(landmark_points_l, axis=0)
    left_cut_plane_center = landmark_points_l_center

    landmark_points_r_center = np.mean(landmark_points_r, axis=0)
    right_cut_plane_center = landmark_points_r_center

    plane_actor_l, plane_center_point_l, plane_normal_l = fitPlaneActorFromPoints(np.array(landmark_points_l), color='green')
    plane_point_l_actor = createSphereActor(plane_center_point_l, radius= 1.0, opacity=1.0, color='red')



    left_cut_plane_center = plane_center_point_l
    left_cut_plane_normal = plane_normal_l

    plane_actor_r, plane_center_point_r, plane_normal_r = fitPlaneActorFromPoints(np.array(landmark_points_r), color='blue')
    plane_point_r_actor = createSphereActor(plane_center_point_r, radius= 1.0, opacity=1.0, color='red')

    right_cut_plane_center = plane_center_point_r
    right_cut_plane_normal = plane_normal_r

    plane_actor_t, plane_center_point_t, plane_normal_t = fitPlaneActorFromPoints(np.array(landmark_points_t), color='red')
    all_actors.append(plane_actor_t)

    # all_actors.append(plane_actor_l)
    # all_actors.append(plane_actor_r)
    #showActors(all_actors)



    bound_points_l, fit_center_point_l, cut_plane_area_l = getClipedCenterPoints(left_cut_plane_center, left_cut_plane_normal, target_poly_data)
    #fit_center_point_actor_l = createSphereActor(fit_center_point_l, radius=1.0, opacity=1.0, color='green')
    #for bound_points_l in bound_points_ls:
    bound_points_actors_l = createPointsActor(bound_points_l, radius=0.2, opacity=1.0, color='green')
    all_actors.extend(bound_points_actors_l)
    #all_actors.append(fit_center_point_actor_l)

    bound_points_r, fit_center_point_r, cut_plane_area_r = getClipedCenterPoints(right_cut_plane_center, right_cut_plane_normal, target_poly_data)
    #fit_center_point_actor_r = createSphereActor(fit_center_point_r, radius=1.0, opacity=1.0, color='blue')

    #for bound_points_r in bound_points_rs:
    bound_points_actors_r = createPointsActor(bound_points_r, radius=0.2, opacity=1.0, color='blue')
    all_actors.extend(bound_points_actors_r)
    #all_actors.append(fit_center_point_actor_r)

    #showActors(all_actors)

    ############### 将切面的中心点，替换之前由4个特征点拟合的中心点   #############

    if np.sqrt(np.sum(np.square(left_cut_plane_center - fit_center_point_l))) <  TwoCentersDistanceThreshold:
        left_cut_plane_center = fit_center_point_l
    if np.sqrt(np.sum(np.square(right_cut_plane_center - fit_center_point_r))) <  TwoCentersDistanceThreshold:
        right_cut_plane_center = fit_center_point_r

    right_center_point_projected_on_left_plane = calProjectedPointCoordOnPlane(left_cut_plane_normal, left_cut_plane_center, right_cut_plane_center)

    left_axis_normalX = normalizeVector(right_center_point_projected_on_left_plane - left_cut_plane_center)
    left_axis_normalY = left_cut_plane_normal
    left_axis_normalZ = np.cross(left_axis_normalX, left_axis_normalY)

    left_center_point_projected_on_right_plane = calProjectedPointCoordOnPlane(right_cut_plane_normal, right_cut_plane_center, left_cut_plane_center)
    right_axis_normalX = normalizeVector(left_center_point_projected_on_right_plane - right_cut_plane_center)
    right_axis_normalY = right_cut_plane_normal
    right_axis_normalZ = np.cross(right_axis_normalX, right_axis_normalY)


    ############ left rotate around X ####################
    left_cut_plane_area_min1, left_bound_points_min1, left_rotate_matrix_min1, left_center_min1 = \
        getTheMinCutPlaneArea(left_axis_normalX, left_axis_normalY, left_cut_plane_center, target_poly_data,
                              dis_threshold=TwoCentersDistanceThreshold)

    if left_bound_points_min1 is not None:
        left_bounds_points_min1_actors = createPointsActor(left_bound_points_min1, radius=0.2, opacity=1.0, color='red')
        all_actors.extend(left_bounds_points_min1_actors)
    ############################################

    ###############  right  roate  around X############
    right_cut_plane_area_min1, right_bound_points_min1, right_rotate_matrix_min1, right_center_min1 = \
        getTheMinCutPlaneArea(right_axis_normalX, right_axis_normalY, right_cut_plane_center, target_poly_data,
                              dis_threshold=TwoCentersDistanceThreshold)

    if right_bound_points_min1 is not None:
        right_bound_points_min1_actors = createPointsActor(right_bound_points_min1, radius=0.2, opacity=1.0, color='red')
        all_actors.extend(right_bound_points_min1_actors)
    ##################################################

    #showActors(all_actors)


    #########  更新椎弓根坐标系和中心点  ###########
    if left_rotate_matrix_min1 is not None:
        left_axis_normalX_new = left_axis_normalX
        left_axis_normalY_new = np.dot(left_axis_normalY, left_rotate_matrix_min1)
        left_axis_normalZ_new = np.dot(left_axis_normalZ, left_rotate_matrix_min1)
        left_cut_plane_center = left_center_min1
    else:
        left_axis_normalX_new = left_axis_normalX
        left_axis_normalY_new = left_axis_normalY
        left_axis_normalZ_new = left_axis_normalZ
    left_cut_plane_normal = left_axis_normalY_new



    if right_rotate_matrix_min1 is not None:
        right_axis_normalX_new = right_axis_normalX
        right_axis_normalY_new = np.dot(right_axis_normalY, right_rotate_matrix_min1)
        right_axis_normalZ_new = np.dot(right_axis_normalZ, right_rotate_matrix_min1)
        right_cut_plane_center = right_center_min1
    else:
        right_axis_normalX_new = right_axis_normalX
        right_axis_normalY_new = right_axis_normalY
        right_axis_normalZ_new = right_axis_normalZ
    right_cut_plane_normal = right_axis_normalY_new



    ############ left rotate around Z ####################
    left_cut_plane_area_min2, left_bound_points_min2, left_rotate_matrix_min2, left_center_min2 = \
        getTheMinCutPlaneArea(left_axis_normalZ_new, left_axis_normalY_new, plane_center_point_l, target_poly_data,
                              dis_threshold=TwoCentersDistanceThreshold)

    if left_bound_points_min2 is not None:
        left_bounds_points_min2_actors = createPointsActor(left_bound_points_min2, radius=0.3, opacity=0.8, color='yellow')
        all_actors.extend(left_bounds_points_min2_actors)
    ######################################################

    ###############  right  roate  around Z############
    right_cut_plane_area_min2, right_bound_points_min2, right_rotate_matrix_min2, right_center_min2 = \
        getTheMinCutPlaneArea(right_axis_normalZ_new, right_axis_normalY_new, plane_center_point_r, target_poly_data,
                              dis_threshold=TwoCentersDistanceThreshold)

    if right_bound_points_min2 is not None:
        right_bound_points_min2_actors = createPointsActor(right_bound_points_min2, radius=0.3, opacity=0.8, color='yellow')
        all_actors.extend(right_bound_points_min2_actors)
    ##################################################

    #showActors(all_actors)

    ##########  step3： update the cut plane center #########
    if right_rotate_matrix_min2 is not None:
        right_axis_normalX_new2 = np.dot(right_axis_normalX_new, right_rotate_matrix_min2)
        right_axis_normalY_new2 = np.dot(right_axis_normalY_new, right_rotate_matrix_min2)
        right_axis_normalZ_new2 = right_axis_normalZ_new
        right_cut_plane_center = right_center_min2
    else:
        right_axis_normalX_new2 = right_axis_normalX_new
        right_axis_normalY_new2 = right_axis_normalY_new
        right_axis_normalZ_new2 = right_axis_normalZ_new

    right_cut_plane_normal = right_axis_normalY_new2



    if left_rotate_matrix_min2 is not None:
        left_cut_plane_center = left_center_min2
        left_axis_normalX_new2 = np.dot(left_axis_normalX_new, left_rotate_matrix_min2)
        left_axis_normalY_new2 = np.dot(left_axis_normalY_new, left_rotate_matrix_min2)
        left_axis_normalZ_new2 = np.dot(left_axis_normalZ_new, left_rotate_matrix_min2)
    else:
        left_axis_normalX_new2 = left_axis_normalX_new
        left_axis_normalY_new2 = left_axis_normalY_new
        left_axis_normalZ_new2 = left_axis_normalZ_new

    left_cut_plane_normal = left_axis_normalY_new2

    ####################  show step3 acotrs  ##########
    # tmp_actors = []
    # tmp_actors.append()
    #showActors(all_actors)
    ###################################################

    ########## step4: 沿着y轴方向搜索最小的截面   ################
    left_cut_plane_area_min3, left_bound_points_min3, left_center_min3 = \
         getTheMinCutPlaneAreaAlongAxisY(left_cut_plane_center, left_cut_plane_normal, target_poly_data)

    right_cut_plane_area_min3, right_bound_points_min3, right_center_min3 = \
        getTheMinCutPlaneAreaAlongAxisY(right_cut_plane_center, right_cut_plane_normal, target_poly_data)


    if np.sqrt(np.sum(np.square(right_center_min3 - right_cut_plane_center))) <  TwoCentersDistanceThreshold:
        right_cut_plane_center = right_center_min3
        right_bound_points_min3_actors = createPointsActor(right_bound_points_min3, radius=0.4, opacity=0.6,
                                                           color='purple')
        all_actors.extend(right_bound_points_min3_actors)

    if np.sqrt(np.sum(np.square(left_center_min3 - left_cut_plane_center))) <  TwoCentersDistanceThreshold:
        left_cut_plane_center = left_center_min3
        left_bounds_points_min3_actors = createPointsActor(left_bound_points_min3, radius=0.4, opacity=0.6,
                                                           color='purple')
        all_actors.extend(left_bounds_points_min3_actors)


    left_cut_plane_center_actor3 = createSphereActor(left_cut_plane_center, radius=0.6, opacity=1.0, color='green')

    right_cut_plane_center_actor3 = createSphereActor(right_cut_plane_center, radius=0.6, opacity=1.0, color='blue')


    ##### check the left and right axis direction #############
    top_plane_center = plane_center_point_t
    tmp = top_plane_center - right_cut_plane_center

    right_flag = np.dot(top_plane_center - right_cut_plane_center, right_axis_normalY_new2)
    if right_flag > 0.0:
        right_axis_normalZ_new2 = -right_axis_normalZ_new2
    else:
        right_axis_normalY_new2 = -right_axis_normalY_new2

    left_flag = np.dot(top_plane_center - left_cut_plane_center, left_axis_normalY_new2)
    if left_flag < 0.0:
        left_axis_normalZ_new2 = -left_axis_normalZ_new2
        left_axis_normalY_new2 = -left_axis_normalY_new2


    ############  add left and right axis  ############
    # right_axis_normalX_actor = createLineActor([right_cut_plane_center, right_cut_plane_center + 15.0 * right_axis_normalX_new2], color='red')
    # right_axis_normalY_actor = createLineActor([right_cut_plane_center, right_cut_plane_center + 10.0 * right_axis_normalY_new2], color='green')
    # right_axis_normalZ_actor = createLineActor([right_cut_plane_center, right_cut_plane_center + 20.0 * right_axis_normalZ_new2], color='blue')
    # all_actors.append(right_axis_normalZ_actor)
    # all_actors.append(right_axis_normalY_actor)
    # all_actors.append(right_axis_normalX_actor)
    #
    # left_axis_normalX_actor = createLineActor([left_cut_plane_center, left_cut_plane_center + 15.0 * left_axis_normalX_new2], color='red')
    # left_axis_normalY_actor = createLineActor([left_cut_plane_center, left_cut_plane_center + 10.0 * left_axis_normalY_new2], color='green')
    # left_axis_normalZ_actor = createLineActor([left_cut_plane_center, left_cut_plane_center + 20.0 * left_axis_normalZ_new2], color='blue')
    #
    # all_actors.append(left_axis_normalX_actor)
    # all_actors.append(left_axis_normalY_actor)
    # all_actors.append(left_axis_normalZ_actor)
    #showActors(all_actors)
    ###################################################



    spine_axis_normalZ = normalizeVector((left_axis_normalZ_new2 + right_axis_normalZ_new2)/2.0)
    spine_axis_center  = (left_cut_plane_center + right_cut_plane_center)/2.0



    left_project_center = calProjectedPointCoordOnPlane(spine_axis_normalZ, spine_axis_center, left_cut_plane_center)
    right_project_center = calProjectedPointCoordOnPlane(spine_axis_normalZ, spine_axis_center, right_cut_plane_center)
    spine_axis_normalY = normalizeVector(left_project_center - right_project_center)
    spine_axis_normalX = vectorCross(spine_axis_normalY, spine_axis_normalZ)
    spine_axis_normalY = spine_axis_normalY

    spine_axis_normalX, spine_axis_normalY = -spine_axis_normalY, spine_axis_normalX

    spine_axis_center_actor = createSphereActor(spine_axis_center, radius=1.0, opacity=1.0, color='red')

    spine_axis_normalX_actor = createLineActor([spine_axis_center, spine_axis_center + 10.0 * spine_axis_normalX], color='red')
    spine_axis_normalY_actor = createLineActor([spine_axis_center, spine_axis_center + 15.0 * spine_axis_normalY], color='green')
    spine_axis_normalZ_actor = createLineActor([spine_axis_center, spine_axis_center + 20.0 * spine_axis_normalZ], color='blue')

    all_actors.append(spine_axis_center_actor)
    all_actors.append(spine_axis_normalX_actor)
    all_actors.append(spine_axis_normalY_actor)
    all_actors.append(spine_axis_normalZ_actor)



    project_point = np.array([0.0, 1.0, 0.0]) ### pedicle pipeline parallel to spine axis z=0 plane

    spine_axis_normal = [spine_axis_normalX, spine_axis_normalY, spine_axis_normalZ]

    pedicle_pipeline_R_normal = createPediclePipelineNormal(15, spine_axis_normal, project_point)
    pedicle_pipeline_points_R = np.array([right_cut_plane_center - 50.0 * pedicle_pipeline_R_normal,
                                          right_cut_plane_center + 50.0 * pedicle_pipeline_R_normal])
    pedicle_pipeline_actor_R = createLineActor(pedicle_pipeline_points_R, color="yellow", line_width=5.0)
    all_actors.append(pedicle_pipeline_actor_R)

    pedicle_pipeline_L_normal = createPediclePipelineNormal(-12, spine_axis_normal, project_point)
    pedicle_pipeline_points_L = np.array([left_cut_plane_center - 50 * pedicle_pipeline_L_normal,
                                          left_cut_plane_center + 50.0 * pedicle_pipeline_L_normal])
    pedicle_pipeline_actor_L = createLineActor(pedicle_pipeline_points_L, color="magenta", line_width=5.0)
    all_actors.append(pedicle_pipeline_actor_L)

    ####  计算通道与椎体的交点   #############

    intersect_points_R = getIntersectPointsFromLineAndPolyData(right_cut_plane_center - 100.0 * pedicle_pipeline_R_normal,
                                                               right_cut_plane_center + 100.0 * pedicle_pipeline_R_normal,
                                                               target_poly_data)

    intersect_points_R_actor = createIntersectPointsActor(intersect_points_R, 1.0, opacity=1.0, color="yellow")


    intersect_points_L = getIntersectPointsFromLineAndPolyData(left_cut_plane_center - 100.0 * pedicle_pipeline_L_normal,
                                                               left_cut_plane_center + 100.0 * pedicle_pipeline_L_normal,
                                                               target_poly_data)
    intersect_points_L_actor = createIntersectPointsActor(intersect_points_L, 1.0, opacity=1.0, color="magenta")


    pedicle_pipeline_cylinder_R_actor = createPediclePipelineCylinderActor(intersect_points_R[0],
                                                                           intersect_points_R[1],
                                                                           color="yellow", )

    save_axis_dir = "E:/data/stl/good_landmark_points/axis"
    save_axis_file = os.path.join(save_axis_dir, window_name.replace(".stl", ".txt"))

    fp = open(save_axis_file, "w")
    fp.write("axis_center:%f,%f,%f\n"%(spine_axis_center[0], spine_axis_center[1], spine_axis_center[2]))
    fp.write("axis_normalX:%f,%f,%f\n"%(spine_axis_normalX[0], spine_axis_normalX[1], spine_axis_normalX[2]))
    fp.write("axis_normalY:%f,%f,%f\n"%(spine_axis_normalY[0], spine_axis_normalY[1], spine_axis_normalY[2]))
    fp.write("axis_normalZ:%f,%f,%f\n"%(spine_axis_normalZ[0], spine_axis_normalZ[1], spine_axis_normalZ[2]))
    fp.write("left cut plane center:%f,%f,%f\n"%(left_cut_plane_center[0], left_cut_plane_center[1], left_cut_plane_center[2]))
    fp.write("right cut plane center:%f,%f,%f\n"%(right_cut_plane_center[0], right_cut_plane_center[1], right_cut_plane_center[2]))
    fp.close()

    #save_coordinate(label, cur_spine_points_center, cur_spine_axis_normal, plane_center_point_L, plane_center_point_R,\
    #  pedicle_pipeline_L_normal, pedicle_pipeline_R_normal)

    pedicle_pipeline_cylinder_L_actor = createPediclePipelineCylinderActor(intersect_points_L[0],
                                                                           intersect_points_L[1],
                                                                           color="magenta", )
    all_actors.extend(intersect_points_R_actor)
    all_actors.extend(intersect_points_L_actor)
    all_actors.append(pedicle_pipeline_cylinder_R_actor)
    all_actors.append(pedicle_pipeline_cylinder_L_actor)

    ###########################################################


    all_actors.append(left_cut_plane_center_actor3)
    all_actors.append(right_cut_plane_center_actor3)




    showActors(all_actors, window_name)

    # save_points_dir = "E:/data/stl/good_landmark_points/landmark"
    # save_point_file = os.path.join(save_points_dir, window_name.replace(".stl", ".txt"))
    # fp = open(save_point_file, "w")
    # for landmark_point in landmark_points_t:
    #     fp.write("%d,%f,%f,%f\n"%(0, landmark_point[0], landmark_point[1], landmark_point[2]))
    #
    # for left_bound_point in left_bound_points_min3:
    #     fp.write("%d,%f,%f,%f\n"%(1, left_bound_point[0], left_bound_point[1], left_bound_point[2]))
    #
    # for right_bound_point in right_bound_points_min3:
    #     fp.write("%d,%f,%f,%f\n"%(2, right_bound_point[0], right_bound_point[1], right_bound_point[2]))
    # fp.close()
    save_png_dir = "E:/data/stl/good_landmark_points/png"
    save_png_file = os.path.join(save_png_dir, window_name.replace(".stl", ".png"))
    saveSpineSurgicalPlanning2Png(all_actors, spine_axis_center, spine_axis_normalZ, spine_axis_normalY, spine_axis_normalX, save_png_file)





def  main():

    STL_SAMPLE_POINTS = 5000
    POINT_NET_INPUT_POINTS = 5000
    # src_label_dir = "/media/hurwa/data3/DeepSpineData/spine_test/Test87_2_yangqiang/predict_SpineV0.6/nrrd"
    # save_label_dir = "/media/hurwa/data3/DeepSpineData/spine_test/Test87_2_yangqiang/predict/predict_landmark"
    # save_stl_dir = "/media/hurwa/data3/DeepSpineData/spine_test/Test87_2_yangqiang/predict_SpineV0.6/stl"
    # os.makedirs(save_label_dir, exist_ok=True)
    # os.makedirs(save_stl_dir, exist_ok=True)


    all_actors = []

    # label_names = os.listdir(src_label_dir)
    # for label_name in label_names:
    #     print(label_name)
    #     src_label_file = os.path.join(src_label_dir, label_name)
    #     itk_img = sitk.ReadImage(src_label_file)
    #     spacing = itk_img.GetSpacing()
    #     origin = itk_img.GetOrigin()
    #     direction = itk_img.GetDirection()
    #
    #     itk_np = sitk.GetArrayFromImage(itk_img)
    #     idx_over0 = np.where(itk_np > 0)
    #
    #     img_np_new = np.zeros_like(itk_np)
    #     img_np_new[idx_over0] = 255
    #
    #
    #
    #     spine_poly_data, _ = createPolyDataNormalsFromArray(img_np_new, spacing, origin, get_largest_connect_region=False)
    #
    #     save_stl_file = os.path.join(save_stl_dir, label_name.replace(".nrrd", ".stl"))
    #     saveSTLFile(save_stl_file, spine_poly_data)
    #     #saveSTLFileFromPolyDataNormals(save_stl_file, spine_poly_data)
    #stl_dir = "E:/data/stl/good_landmark_points/stl_train"
    stl_dir = "E:/data/DeepSpineData/stl_hard"
    stl_names = os.listdir(stl_dir)

    for stl_name in stl_names:
        label = stl_name.split("_")[-1][0:2]
        # label = int(label)
        # if "origin" in stl_name:
        #     continue
        # if "rot" in stl_name:
        #     continue
        # if "Test78_1_seg_label_17" not in stl_name:
        #     continue
        if "new" not in stl_name:
            continue
        # if "origin_rotz.stl" not in stl_name:
        #     continue
        print(stl_name)
        stl_file = os.path.join(stl_dir, stl_name)

        spine_actor = createActorFromSTL(stl_file)

        spine_points = getPointsFromSTL(stl_file, num_points=2*POINT_NET_INPUT_POINTS)

        ##################### show sample points  ##################

        # ply = o3d.io.read_triangle_mesh(stl_file)
        # colors = np.array([255.0, 0.0, 0.0]) / 255 * np.ones((STL_SAMPLE_POINTS, 3), dtype=np.float32)
        #
        # #pcd = ply.sample_points_uniformly(number_of_points = STL_SAMPLE_POINTS)
        # pcd = ply.sample_points_poisson_disk(number_of_points=STL_SAMPLE_POINTS)
        #
        # pcd_np = np.asarray(pcd.points)
        #
        # points = o3d.utility.Vector3dVector(pcd_np)
        # colors = o3d.utility.Vector3dVector(colors)
        # pcd.points = points
        # pcd.colors = colors
        #
        # o3d.visualization.draw_geometries([pcd], window_name='%s' % stl_name)

        ####################################################

        choice = np.random.choice(spine_points.shape[0], POINT_NET_INPUT_POINTS, replace=False)
        spine_points = spine_points[choice, :]




        model_dir = "./"
        spine_points_label = predict(spine_points, model_dir=model_dir)


        index_top = np.where(spine_points_label == 1)
        index_left = np.where(spine_points_label == 2)
        index_right = np.where(spine_points_label == 3)


        top_points = spine_points[index_top, :][0]
        left_points = spine_points[index_left, :][0]
        right_points = spine_points[index_right, :][0]

        top_points_actor = createPointsActor(top_points, radius=0.5, opacity=1.0, color="red")
        left_points_actor = createPointsActor(left_points, radius=0.5, opacity=1.0, color='green')
        right_points_actor = createPointsActor(right_points, radius=0.5, opacity=1.0, color='blue')
        #spine_points_actors = createPointsActor(spine_points, radius=0.2, opacity=1.0, color='red')
        all_actors = []
        all_actors.append(spine_actor)
        all_actors.extend(top_points_actor)
        all_actors.extend(left_points_actor)
        all_actors.extend(right_points_actor)
        #all_actors.extend(spine_points_actors)

        showActors(all_actors)

        target_poly_data = createPolyDataFromSTL(stl_file)

        PedicleSurgeryPlanning(top_points, left_points, right_points, target_poly_data, window_name=stl_name)
        #break


def createLandmarkPointsFromPickedPoints():
    stl_dir = "E:/data/stl/good_landmark_points/stl_test"
    picked_points_dir = "E:/data/stl/good_landmark_points/picked_points"

    landmark_names = os.listdir(picked_points_dir)
    for landmark_name in landmark_names:
        if "new_Spine_seg_label_09" not in landmark_name:
            continue
        landmark_file = os.path.join(picked_points_dir, landmark_name)
        landmark_points = parsePointsFile(landmark_file)
        stl_name = landmark_name.replace("_picked_points.pp", ".stl")

        stl_file = os.path.join(stl_dir, stl_name)
        target_poly_data = createPolyDataFromSTL(stl_file)

        print(stl_name)
        top_points = []
        left_points = []
        right_points = []
        for k, v in landmark_points.items():
            if "L" in k:
                left_points.append(v)
            if "R" in k:
                right_points.append(v)
            if "T" in k:
                top_points.append(v)
        top_points = np.array(top_points)
        left_points = np.array(left_points)
        right_points = np.array(right_points)

        target_actor = createActorFromPolydata(target_poly_data, opacity=0.8)
        top_points_actors = createPointsActor(top_points, radius=1.0, opacity=1.0, color='red')
        left_points_actors = createPointsActor(left_points, radius=1.0, opacity=1.0, color='green')
        right_points_actors = createPointsActor(right_points, radius=1.0, opacity=1.0, color='blue')
        all_actors = []
        all_actors.append(target_actor)
        all_actors.extend(top_points_actors)
        all_actors.extend(left_points_actors)
        all_actors.extend(right_points_actors)
        showActors(all_actors)

        PedicleSurgeryPlanning(top_points, left_points, right_points, target_poly_data, window_name=stl_name)


if __name__ == "__main__":
    main()
    #createLandmarkPointsFromPickedPoints()