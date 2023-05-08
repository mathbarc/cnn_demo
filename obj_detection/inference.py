import cv2
import numpy
import time


def sigmoid(data):
    return 1/(1+(numpy.exp(-data)))

def silu(data):
    return data * sigmoid(data)

def softmax(data):
    return numpy.exp(data) / numpy.sum(numpy.exp(data))

labels = ["crop", "weed"]
net = cv2.dnn.readNetFromONNX("object_detection_last.onnx")
n_objects_per_cell = 5

flops = net.getFLOPS((1, 3, 512, 512)) * 10e-9
print(round(flops, 3), "BFLOPs")

img = cv2.imread("obj_detection/dataset/./agri_data/data/agri_0_9712.jpeg")
input_img = cv2.dnn.blobFromImage(
    img, scalefactor=1.0 / 255.0, size=(512, 512), swapRB=True
)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output").transpose((0, 2, 3, 1))


objs = output.reshape((-1, output.shape[-1]))
objs = output.reshape(
    (int(objs.shape[0] * n_objects_per_cell), int(objs.shape[1] / n_objects_per_cell))
)

# objs = 1/(1+numpy.exp(objs))

boxes = [
    (
        int(img.shape[1] * sigmoid(obj[0])),
        int(img.shape[0] * sigmoid(obj[1])),
        
        int(img.shape[1] * (sigmoid(obj[2]) - sigmoid(obj[0]))),
        int(img.shape[0] * (sigmoid(obj[3]) - sigmoid(obj[1]))),
    )
    for obj in objs
]
classes = [softmax(obj[5:]) for obj in objs]
prob = [sigmoid(obj[4]) for obj in objs]
indexes = cv2.dnn.NMSBoxes(boxes, prob, 0.3, 0.4)
end = time.time()
print(end - start)



for box_id in indexes:
    
    label_id = classes[box_id].argmax()
    label = labels[label_id]
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
