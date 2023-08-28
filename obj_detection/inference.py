import cv2
import numpy
import time


def sigmoid(data):
    return 1/(1+(numpy.exp(-data)))

def silu(data):
    return data * sigmoid(data)

def softmax(data):
    exp_data = numpy.exp(data)
    sums = numpy.sum(exp_data,axis=1)
    for i in range(exp_data.shape[0]):
        exp_data[i,:] = exp_data[i,:]/sums[i]
    return exp_data 


labels_str = ["crop", "weed"]
net = cv2.dnn.readNetFromONNX("object_detection_best.onnx")
# net = cv2.dnn.readNetFromDarknet("yolov2-tiny.cfg","yolov2-tiny.weights")

# net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

flops = net.getFLOPS((1, 3, 512, 512)) * 10e-9
print(round(flops, 3), "BFLOPs")

img = cv2.imread("obj_detection/dataset/./agri_data/data/agri_0_3.jpeg")
input_img = cv2.dnn.blobFromImage(
    img, scalefactor=1.0 / 255.0, size=(512, 512), swapRB=True
)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output")

boxes = []
classes = []
prob = []

for b in range(output.shape[1]):
    
    boxes.append(output[0,b,:4].astype(int))
    classes.append(output[0,b,5:])
    prob.append(output[0,b,4])



indexes = cv2.dnn.NMSBoxes(boxes, prob, 0.8, 0.1)
end = time.time()
print(end - start)



for box_id in indexes:
    
    label_id = classes[box_id].argmax()
    label = labels_str[label_id]
    box_classes = classes[box_id]

    cv2.rectangle(img, boxes[box_id], (0, 255, 0), 3)
    class_prob = str(round(box_classes[label_id], 2))
    box_prob = str(round(prob[box_id], 2))
    cv2.putText(
        img,
        f"{label} - {class_prob}, {box_prob}",
        boxes[box_id][0:2],
        cv2.FONT_HERSHEY_PLAIN,
        2,
        (255, 0, 0),
        2,
    )
    print(boxes[box_id])
    print(box_classes, "->", label)
    print(prob[box_id])

cv2.imshow("img", img)
cv2.imwrite("pred.png", img)
cv2.waitKey(0)
