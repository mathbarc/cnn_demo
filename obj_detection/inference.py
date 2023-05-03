import cv2
import numpy
import time

net = cv2.dnn.readNetFromONNX("object_detection_1000.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print(net.getFLOPS((1,3,512,512)))

img = cv2.imread("obj_detection/dataset/agri_data/data/agri_0_3.jpeg")
input_img = cv2.dnn.blobFromImage(img, 1./.255,(512,512), swapRB=True)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output").transpose((0,2,3,1))
end = time.time()
print(end-start)

objs = output.reshape((-1,output.shape[-1]))
boxes = [obj[0:4] for obj in objs if obj[4]>0.5]
classes = [obj[5:] for obj in objs if obj[4]>0.5]
prob = [obj[4] for obj in objs if obj[4]>0.5]

cv2.imshow("img", img)
cv2.waitKey(0)

...