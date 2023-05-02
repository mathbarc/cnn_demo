import cv2
import time

net = cv2.dnn.readNetFromONNX("object_detection_2000.onnx")

print(net.getFLOPS((1,3,512,512)))

img = cv2.imread("obj_detection/dataset/agri_data/data/agri_0_76.jpeg")
input_img = cv2.dnn.blobFromImage(img, 1./.255,(512,512), swapRB=True)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output")
end = time.time()
print(end-start)

cv2.imshow("img", img)
cv2.waitKey(0)

...