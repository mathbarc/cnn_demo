import cv2
import numpy
import time

labels =["crop", "weed"]
net = cv2.dnn.readNetFromONNX("object_detection_2000.onnx")

print(net.getFLOPS((1,3,512,512))*10e-9, "BFLOPs")

img = cv2.imread("obj_detection/dataset/./agri_data/data/agri_0_6258.jpeg")
input_img = cv2.dnn.blobFromImage(img, scalefactor=1./255.,size=(512,512), swapRB=True)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output").transpose((0,2,3,1))
end = time.time()
print(end-start)

objs = output.reshape((-1,output.shape[-1]))
objs = output.reshape((int(objs.shape[0]*3),int(objs.shape[1]/3)))

# objs = 1/(1+numpy.exp(objs))

boxes = [ (int(img.shape[1]*obj[0]), int(img.shape[0]*obj[1]), int(img.shape[1]*(obj[2]-obj[0])), int(img.shape[0]*(obj[3]-obj[1]))) for obj in objs]
classes = [obj[5:] for obj in objs]
prob = [obj[4] for obj in objs]

indexes = cv2.dnn.NMSBoxes(boxes, prob, 0.5, 0.4)

for box_id in indexes:
    cv2.rectangle(img, boxes[box_id], (0,255,0),3)
    print(boxes[box_id])
    print(classes[box_id],"->", labels[numpy.argmax(classes[box_id])])
    print(prob[box_id])

cv2.imshow("img", img)
cv2.waitKey(0)
