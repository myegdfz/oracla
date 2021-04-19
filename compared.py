
import cv2
import numpy as np

def sifts(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    sift = cv2.xfeatures2d.SIFT_create()

    kp, des = sift.detectAndCompute(gray, None)
    if type(des) == type(None):
        return des
    eps = 1e-7
    des /= (des.sum(axis=1, keepdims=True) + eps)
    des = np.sqrt(des)
    # des/=(np.linalg.norm(des,axis=1,ord=2)+eps)#可选选项

    return des

