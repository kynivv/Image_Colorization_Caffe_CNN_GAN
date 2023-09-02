# Libraries
import numpy as np
import cv2
from cv2 import dnn


# Models Paths
proto_file = 'models/colorization_deploy_v2.prototxt'

model_file = 'models/colorization_release_v2.caffemodel'

hull_pts = 'models/pts_in_hull.npy'


# Img Path
img_path = 'test_img/img.jpg'


# Reading PreTrained Model Parameters
net = dnn.readNetFromCaffe(proto_file, model_file)
kernel = np.load(hull_pts)


# Img Preprocessing
img = cv2.imread(img_path)
scaled = img.astype('float32') / 255.0
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)


# Cluster Center as 1x1 Convolutions To the Model
class8 = net.getLayerId('class8_ab')
conv8 = net.getLayerId('conv8_313_rh')
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype('float32')]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]


# Img Resize
resized = cv2.resize(lab_img, (224, 224))


# Split L Channel
L = cv2.split(resized)[0]


# Mean Subtraction
L -= 50 


# Predicting the AB Channels From the Input L Channel
net.setInput(cv2.dnn.blobFromImage(L))

ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))


# Resize Predicted "ab" Volume To the Same Dimensions as Our Input Image
ab_channel = cv2.resize(ab_channel, (img.shape[0], img.shape[1]))


# Take the L Channel From Image
L = cv2.split(lab_img)[0]


# Join the L Channel With Our Predicted "ab" Channel
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis= 2)


# Converting Image From LAB To BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)


# Changing Img Range to 255 And Convering It From Float32 To INT
colorized = (255 * colorized).astype('uint8')


# Displaying Input And Output Images
img = cv2.resize(img, (700, 700))
colorized = cv2.resize(colorized, (700, 700))

result = cv2.hconcat([img, colorized])

cv2.imshow('Input And Output', result)

cv2.waitKey(0)
