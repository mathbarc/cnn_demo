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
anchors = numpy.array([[47, 43],  [94,105], [210,207], [361,283], [442,425]],dtype=float)*(1/512)
net = cv2.dnn.readNetFromONNX("object_detection_last.onnx")

n_objects_per_cell = 5

flops = net.getFLOPS((1, 3, 512, 512)) * 10e-9
print(round(flops, 3), "BFLOPs")

img = cv2.imread("obj_detection/dataset/./agri_data/data/agri_0_3.jpeg")
input_img = cv2.dnn.blobFromImage(
    img, scalefactor=1.0 / 255.0, size=(512, 512), swapRB=True
)

start = time.time()
net.setInput(input_img, "features")
output = net.forward("output")
output = output.reshape((output.shape[0],n_objects_per_cell,int(output.shape[1]/n_objects_per_cell), output.shape[2], output.shape[3] ))

boxes = []
classes = []
prob = []



for y in range(output.shape[3]):
    for x in range(output.shape[4]):
        box = output[:,:,:4,y,x].reshape((n_objects_per_cell,4))
        objness = output[:,:,4,y,x].reshape((n_objects_per_cell))
        labels = output[:,:,5:,y,x].reshape((n_objects_per_cell,output.shape[2]-5))

        box[:,0] = ((x + sigmoid(box[:,0]))/output.shape[4])*img.shape[1]
        box[:,1] = ((y + sigmoid(box[:,1]))/output.shape[3])*img.shape[0]
        box[:,2] = (numpy.exp(box[:,2])*anchors[:,0])*img.shape[1]
        box[:,3] = (numpy.exp(box[:,3])*anchors[:,1])*img.shape[0]

        box[:,0] = box[:,0] - (box[:,2]*0.5)
        box[:,1] = box[:,1] - (box[:,3]*0.5)

        

        objness = sigmoid(objness)
        labels = softmax(labels)

        for obj_i in range(box.shape[0]):
            boxes.append(box[obj_i,:].astype(int))
            classes.append(labels[obj_i,:])
            prob.append(objness[obj_i])





# objs = output.reshape((-1, output.shape[-1]))
# objs = output.reshape(
#     (int(objs.shape[0] * n_objects_per_cell), int(objs.shape[1] / n_objects_per_cell))
# )

# objs = 1/(1+numpy.exp(objs))


indexes = cv2.dnn.NMSBoxes(boxes, prob, 0.4, 0.4)
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
