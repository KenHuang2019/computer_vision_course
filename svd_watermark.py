import cv2
import numpy as np
from numpy import linalg as la

def replace_sv(s1, s2):
    """
    concate last half of sigma 2 (s2) to first half of sigma 1 (s1) 
    c_sigma = [s1 + s2]
    """
    n = int(len(s1) / 2)
    c = np.concatenate((s1[:n], s2[n:]/10), axis=None) # /10 as a compression

    # convert vector to metrix
    S = np.zeros([len(s1),len(s1)]) 
    for i in range(len(s1)):
        S[i][i] = c[i]

    return  S

if __name__ == '__main__':
    A_img = cv2.imread("./input/imitation_game_gray.jpg", cv2.IMREAD_GRAYSCALE)
    B_img = cv2.imread("./input/imitation_game_2_gray.jpg", cv2.IMREAD_GRAYSCALE)

    h, w = A_img.shape
    print("Original img size: ", h, w)

    scale =  1
    r_A = cv2.resize(A_img, ( int(w/scale), int(h/scale)), 1)
    r_B = cv2.resize(B_img, ( int(w/scale), int(h/scale)), 1)

    r_h, r_w = r_A.shape
    print("Resized img size: ", r_h, r_w)

    U_A,sigma_A,VT_A = la.svd(r_A, full_matrices=False)
    U_B,sigma_B,VT_B = la.svd(r_B, full_matrices=False)
    print("sigma_A: ",sigma_A)

    concated_sigma_metrix = replace_sv(sigma_A, sigma_B)

    # U_A * c_sigma * VT_A
    t = np.dot(U_A, concated_sigma_metrix)
    watermarked_img = np.dot(t, VT_A)

    cv2.imwrite('./output/'+'SVD_A'+'.jpg', watermarked_img)