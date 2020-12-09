import dlib
from skimage import io
import numpy as np
import os

def saveMask(img, attr_points, maskarray, patch_size, filename):
    sum_x = 0
    sum_y = 0
    num_points = maskarray[-1]-maskarray[0]+1
    for i in range(maskarray[0],maskarray[-1]+1):
        # print("shape:",shape[i].x,shape[i].x)
        sum_x += attr_points[i].x
        sum_y += attr_points[i].y

    # img[50:100,50:100] # 50~100 行，50~100 列（不包括第 100 行和第 100 列）
    l_x = int(sum_x/num_points-patch_size/2)
    r_x = int(sum_x/num_points+patch_size/2)
    t_y = int(sum_y/num_points-patch_size/2)
    b_y = int(sum_y/num_points+patch_size/2)
    cropped = img[t_y:b_y, l_x:r_x]  
    io.imsave(filename,cropped)
    # plt.imshow(cropped)

def work(output_dir, data_dir, imgname):
    fname = os.path.splitext(imgname)[0]
    ftype = os.path.splitext(imgname)[1]
    output_dir = os.path.join(output_dir, fname)
    # 使用68点特征提取器
    predictor_path = "shape_predictor_68_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    # 读取图片
    img = io.imread(os.path.join(data_dir, imgname))
    # plt.imshow(img)

    # 扫描的人脸区域，用索引区分
    faces = detector(img, 1)
    # print("faces:",faces[0])

    # 按68点定位取中点，往两边扩展截取
    l_eye_idx = [36,41]
    r_eye_idx = [42,47]
    nose_idx = [27,35]
    # 截取的区域大小
    patch_size = img.shape[0]/8

    for k, d in enumerate(faces):
        attr_points = predictor(img, d)
        attr_points = np.array(attr_points.parts())
        saveMask(img, attr_points, l_eye_idx, patch_size, output_dir+'_leye'+ftype)
        saveMask(img, attr_points, r_eye_idx, patch_size, output_dir+'_reye'+ftype)
        saveMask(img, attr_points, nose_idx, patch_size, output_dir+'_nose'+ftype)
if __name__ == '__main__':
    data_dir = 'pics'
    output_dir = 'output'
    print("Start the Process")
    for f in os.listdir(data_dir):
        if f.endswith(".jpg"):
            work(output_dir, data_dir, f)
            print("Finished the file:",f)
    print("Finished the Process")

