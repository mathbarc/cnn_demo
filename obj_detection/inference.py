import cv2
import numpy
import time

net = cv2.dnn.readNetFromONNX("object_detection_1000.onnx")

print(net.getFLOPS((1,3,512,512)))

img = cv2.imread("obj_detection/dataset/agri_data/data/agri_0_3.jpeg")
input_img = cv2.dnn.blobFromImage(img, scalefactor=1./255,size=(512,512), swapRB=True)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output").transpose((0,2,3,1))
end = time.time()
print(end-start)

objs = output.reshape((-1,output.shape[-1]))
objs = output.reshape((int(objs.shape[0]*3),int(objs.shape[1]/3)))
boxes = [obj[0:4] for obj in objs]
classes = [obj[5:] for obj in objs]
prob = [obj[4] for obj in objs]

cv2.imshow("img", img)
cv2.waitKey(0)

...