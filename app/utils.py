import os
import numpy as np
import tensorflow as tf
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image

def preprocess(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    return img

def load_image(image_path):
  im = Image.open(image_path)
  return np.array(im)

def extract_face(img):

  #img = cv2.imread(img_path)
  detector = MTCNN()
  faces = detector.detect_faces(img)

  #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
  x1,y1,w,h = faces[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2 = abs(x1+w)
  y2 = abs(y1+h)

  #locate the co-ordinates of face in the image
  store_face = img[y1:y2,x1:x2]
  store_face = cv2.resize(store_face, (224, 224)) #The VGGFace model expects a 224x224x3 size face image as input, and it outputs a face embedding vector with a length of 2048.
  return store_face

def verify(model, detection_threshold, verification_threshold):
  results = []
  inp_img = "../application_data/input_image/input_image.jpg"
  verif_folder = "../application_data/verification_images"
  for image in os.listdir(verif_folder):
    input_img = preprocess(inp_img)
    validation_img = preprocess(os.path.join(verif_folder,image))
    result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
    results.append(result)
  detection = np.sum(np.array(results) > detection_threshold)
  verification = detection / len(os.listdir(verif_folder))
  verified = verification > verification_threshold
  return results, verified