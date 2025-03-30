import cv2
import numpy
import time
import os
import glob


def sigmoid(data):
    return 1 / (1 + (numpy.exp(-data)))


def silu(data):
    return data * sigmoid(data)


def softmax(data):
    exp_data = numpy.exp(data)
    sums = numpy.sum(exp_data, axis=1)
    for i in range(exp_data.shape[0]):
        exp_data[i, :] = exp_data[i, :] / sums[i]
    return exp_data


if __name__ == "__main__":
    # net = cv2.dnn.readNetFromONNX("yolo11n.onnx")
    net = cv2.dnn.readNetFromONNX("obj_detection_last.onnx")
    # net = cv2.dnn.readNetFromONNX("last.onnx")
    # net = cv2.dnn.readNetFromONNX("obj_detection_last.onnx")
    # net = cv2.dnn.readNetFromTorch("obj_det.pt")
    # net = cv2.dnn.readNetFromDarknet("yolov2-tiny.cfg","yolov2-tiny.weights")

    obj_threshold = 0.4
    nms_threshold = 0.4

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    flops = net.getFLOPS((1, 3, 416, 416)) * 10e-9
    print(round(flops, 3), "BFLOPs")

    # input_test = numpy.ones((1, 3, 416, 416))*0.5

    # mean = 0
    # for i in range(10):

    #     start = time.time()
    #     net.setInput(input_test, "features")
    #     output = net.forward(["output"])
    #     end = time.time()
    #     v = (end - start)
    #     if i:
    #         mean += v
    #     print(v)
    #     time.sleep(1)

    # print()
    # mean = mean/9
    # print(mean, "s", 1/mean, "fps")
    # ...

    # img = cv2.imread("/data/hd1/Dataset/Coco/images/000000391895.jpg")
    labels_str = [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    # folder = "/data/hd1/Dataset/leafs/images/"
    folder = "/data/ssd1/Datasets_old/Coco/test2017/"
    output_folder = "./inference_results"

    os.makedirs(output_folder, exist_ok=True)

    outLayers = net.getUnconnectedOutLayersNames()
    print(outLayers)

    count = 0
    mean_infer_time = 0
    mean_post_time = 0
    for file_path in glob.glob("*.jpg", root_dir=folder):
        file = os.path.join(folder, file_path)
        filename = os.path.basename(file_path).split(".")[0]
        img = cv2.imread(file)

        input_img = cv2.dnn.blobFromImage(
            img, scalefactor=1.0 / 255.0, size=(416, 416), swapRB=True
        )

        start = time.time()
        net.setInput(input_img, "features")
        output = net.forward(outLayers)[0]
        finished_inference = time.time()

        boxes = []
        classes = []
        prob = []

        # output = numpy.transpose(output, (0,1,3,4,2))
        output = numpy.reshape(
            output,
            (
                output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3],
                output.shape[4],
            ),
        )

        for b in range(output.shape[0]):
            box = output[b, :4].copy()
            box[0] = (box[0] - box[2] * 0.5) * img.shape[1]
            box[1] = (box[1] - box[3] * 0.5) * img.shape[0]
            box[2] = box[2] * img.shape[1]
            box[3] = box[3] * img.shape[0]

            boxes.append(box.astype(int))

            cl = output[b, 5:]
            classes.append(cl)
            prob.append(output[b, 4])

        indexes = cv2.dnn.NMSBoxes(boxes, prob, obj_threshold, nms_threshold)
        finished_output_postprocess = time.time()

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
            orig_box = output[box_id, :4].copy()
            orig_box[0] *= img.shape[1]
            orig_box[1] *= img.shape[0]
            orig_box[2] *= img.shape[1]
            orig_box[3] *= img.shape[0]
            print(boxes[box_id], "->", orig_box)
            print(box_classes, "->", label)
            print(prob[box_id])

        cv2.imwrite(f"{output_folder}/pred_{filename}.png", img)

        infer_time = finished_inference - start
        mean_infer_time += infer_time

        post_time = finished_output_postprocess - start
        mean_post_time += post_time
        count += 1

        print(file_path, infer_time, "s", post_time, "s")

        # cv2.imshow("img", img)
        # cv2.waitKey(0)

    print("mean inference time: ", mean_infer_time / count, "s")
    print("mean postprocess time: ", mean_post_time / count, "s")
