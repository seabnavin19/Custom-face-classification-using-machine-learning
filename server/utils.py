import pickle
import json
import joblib
from wavelet import w2d
import numpy as np
import cv2
import base64
__model=None
def get_crop(bs4):
    face_cascade = cv2.CascadeClassifier("../artifacts/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("../artifacts/haarcascade_eye.xml")
    img = get_cv2_image_from_base64_string(bs4)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    croped_face=[]
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes) >= 2:
            croped_face.append(roi_color)
            print(roi_color)
    return croped_face
def v_stack(bs4):
    imgs=get_crop(bs4)
    result=[]
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32 * 32 * 3 + 32 * 32

        final = combined_img.reshape(1, len_image_array).astype(float)
        result.append(__model.predict(final)[0])
    return result
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_save():
    global __model
    with open('../artifacts/model.pkl', 'rb') as f:
        __model = joblib.load(f)
def get_b64_test_image_for_virat():
    with open("b64.txt") as f:
        return f.read()

if __name__=="__main__":
    load_save()
    # print(v_stack("/home/kit/Projects/image/model/Dataset/lionel_messi/8e28dca199d2c529e710f2fc7550fc85.jpg"))
    # print(v_stack(get_b64_test_image_for_virat()))
