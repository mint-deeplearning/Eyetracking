import numpy as np
import cv2
import glob
import pickle
import os

def camera_calibration(images_path, chessboard_size=(9,6), square_size=0.025):

    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  
    
    
    objpoints = []  
    imgpoints = []  
    

    images = glob.glob(images_path)
    
    if len(images) == 0:
        print("未找到校准图像！")
        return None, None, None
    
    print(f"找到 {len(images)} 张校准图像")
    

    for i, fname in enumerate(images):
        print(f"处理图像 {i+1}/{len(images)}: {fname}")
        
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        

        if ret:
            objpoints.append(objp)
            

            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners_refined)
            
            img_draw = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners_refined, ret)
            cv2.imshow('Chessboard Corners', img_draw)
            cv2.waitKey(500)  
        else:
            print(f"未在图像 {fname} 中找到棋盘格角点")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) == 0:
        print("未在任何图像中找到有效的棋盘格角点！")
        return None, None, None
    
    print(f"成功处理 {len(objpoints)} 张图像进行校准")
    

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
  
    return camera_matrix, dist_coeffs, objpoints, imgpoints

def save_calibration_results(filename, camera_matrix, dist_coeffs):
    """保存校准结果到文件"""
    calibration_data = {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    }
    
    with open(filename, 'wb') as f:
        pickle.dump(calibration_data, f)

def load_calibration_results(filename):
    """从文件加载校准结果"""
    with open(filename, 'rb') as f:
        calibration_data = pickle.load(f)
    
    camera_matrix = calibration_data['camera_matrix']
    dist_coeffs = calibration_data['dist_coeffs']

    return camera_matrix, dist_coeffs
