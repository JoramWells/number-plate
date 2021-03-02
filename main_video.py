# import the necessary packages
import numpy as np
import argparse
import cv2
from imutils import paths
import time
import imutils
from imutils.video import VideoStream

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--folder", required=True,
#                 help="path to folder image")
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-o", "--output", type=str,
                help="path to output video")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

while True:
    # grab the frame from the threaded video stream
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    frame = vs.read()

    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > args["confidence"]:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print("[INFO] {}".format(label))
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        if writer is None and args["output"] is not None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 20,
                                     (frame.shape[1], frame.shape[0]), True)
        # if the writer is not None, write the frame with recognized
        # faces to disk
        if writer is not None:
            writer.write(frame)

    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # rgb = imutils.resize(frame, width=750)
    # r = frame.shape[1] / float(rgb.shape[1])
    # # detect the (x, y)-coordinates of the bounding boxes
    # # corresponding to each face in the input frame, then compute
    # # the facial embeddings for each face
    # boxes = face_recognition.face_locations(rgb,
    #                                         model=args["detection_method"])
    # encodings = face_recognition.face_encodings(rgb, boxes)
    # names = []

# print("[INFO] loading model...")
# net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# folder = sorted(list(paths.list_images(args["folder"])))


# for image in folder:
#     image = cv2.imread(image)
#     (h, w) = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
#                                  (300, 300), 127.5)

    # print("[INFO] computing object detections...")
    # net.setInput(blob)
    # detections = net.forward()

    # for i in np.arange(0, detections.shape[2]):
    #     # extract the confidence (i.e., probability) associated with the
    #     # prediction
    #     confidence = detections[0, 0, i, 2]
    #     # filter out weak detections by ensuring the `confidence` is
    #     # greater than the minimum confidence
    #     if confidence > args["confidence"]:
    #         # extract the index of the class label from the `detections`,
    #         # then compute the (x, y)-coordinates of the bounding box for
    #         # the object
    #         idx = int(detections[0, 0, i, 1])
    #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    #         (startX, startY, endX, endY) = box.astype("int")
    #         # display the prediction
    #         label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
    #         print("[INFO] {}".format(label))
    #         cv2.rectangle(image, (startX, startY), (endX, endY),
    #                       COLORS[idx], 2)
    #         y = startY - 15 if startY - 15 > 15 else startY + 15
    #         cv2.putText(image, label, (startX, y),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show the output image
    cv2.imshow("Output", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    vs.stop()
