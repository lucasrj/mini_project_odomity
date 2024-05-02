import cv2
import numpy as np
import matplotlib.pyplot as plt

import src.visual_slam as vs   


def get_orb(image1, image2):
    
    # Detect and show sift features
    ORB = cv2.ORB_create()
    kp_image1, des_image1 = ORB.detectAndCompute(image1, None)
    kp_image2, des_image2 = ORB.detectAndCompute(image2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des_image1, des_image2)

    points1_temp = []
    points2_temp = []
    match_indices_temp = []

    for idx, m in enumerate(matches):
        points1_temp.append(kp_image1[m.queryIdx].pt)
        points2_temp.append(kp_image2[m.trainIdx].pt)
        match_indices_temp.append(idx)

    points1 = np.float32(points1_temp)
    points2 = np.float32(points2_temp)
    # match_indices = np.int32(match_indices_temp)
    ransacReprojecThreshold = 1
    confidence = 0.99
    cameraMatrix = get_camera_matrix()

    # Remember that points1 and point2 should be floats.
    essentialMatrix, mask = cv2.findEssentialMat(
            points1, 
            points2, 
            cameraMatrix,
            cv2.FM_RANSAC, 
            confidence,
            ransacReprojecThreshold) 

    return essentialMatrix, points1, points2


def calc_dist(self,KeyPs,epipolLines):
    dist = []
    for pt, line in zip(KeyPs, epipolLines):
         x = pt[0]; y = pt[1]
         a = line[0]; b = line[1]; c = line[2]
            
    dst = (a*x + b*y + c) / np.sqrt(a**2 + b**2)
    dist.append(dst)

    return dist

def get_camera_matrix(self):
        k = np.array([[2676, 0., 3840 / 2 - 35.24], 
            [0.000000000000e+00, 2676., 2160 / 2 - 279],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        return k

def calc_epi_dist(self,essentialMatrix,feature1,feature2):
        
        epiLine1 = cv2.computeCorrespondEpilines(feature1.reshape(-1, 1, 2), 2, essentialMatrix)
        epiLine1 = epiLine1.reshape(-1,3)
        epiLine2 = cv2.computeCorrespondEpilines(feature2.reshape(-1, 1, 2), 2, essentialMatrix)
        epiLine2 = epiLine2.reshape(-1,3)

        dist1 = self.calc_dist(feature1,epiLine1)
        dist2 = self.calc_dist(feature2,epiLine2)

        print('mean dist 1: ', np.mean(dist1))
        print('std deviation dist 1: ', np.std(dist1))

        print('mean dist 2: ', np.mean(dist2))
        print('std deviation dist 2: ', np.std(dist2))

        print('mean: ', np.mean(dist1))
        print('std deviation: ', np.std(dist1))
        plt.hist(dist1, bins=200)
        plt.show()

        print('mean: ', np.mean(dist2))
        print('std deviation: ', np.std(dist2))
        plt.hist(dist2, bins=200)
        plt.show()


if __name__ == '__main__':
    
    img1 = cv2.imread("/home/oliver/Documents/LargeScale_Drone/Miniproject2/input/frames/frame001450.jpg")
    img2 = cv2.imread("/home/oliver/Documents/LargeScale_Drone/Miniproject2/input/frames/frame001525.jpg")

    essentialMatrix, pts1, pts2 = get_orb(img1, img2)
    calc_epi_dist( essentialMatrix, pts1, pts2)
    
    # Calculates relative movement
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    frameGen = vs.FrameGenerator(cv2.cv2.ORB_create())
    vs.current_image_pair = vs.ImagePair(img1, img2, bf, get_camera_matrix())
    vs.current_image_pair.match_features()
    essential_matches = vs.current_image_pair.determine_essential_matrix(vs.current_image_pair.filtered_matches)
    vs.current_image_pair.estimate_camera_movement(essential_matches)

