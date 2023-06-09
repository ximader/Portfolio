#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 12:51:15 2023

@author:    Alexander Lomakin
@credits:   Yuichiro Tachibana (https://github.com/whitphx/streamlit-webrtc-example)
            Randy Zwitch (https://github.com/streamlit/demo-self-driving)

"""

import os
import urllib
import av
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "opencv_face_detector_uint8.pb": {
        "url": "https://github.com/ximader/Portfolio/raw/main/pet_projects/webcam_age_recognition_with_streamlit/opencv_face_detector_uint8.pb",
        "size": 2727750
    },
    "opencv_face_detector.pbtxt": {
        "url": "https://github.com/ximader/Portfolio/raw/main/pet_projects/webcam_age_recognition_with_streamlit/opencv_face_detector.pbtxt",
        "size": 37272
    },
    "age_from_face_model.tflite": {
        "url": "https://github.com/ximader/Portfolio/raw/main/pet_projects/webcam_age_recognition_with_streamlit/age_from_face_model.tflite",
        "size": 23935320
    }
}


# GLOBALS 
faceModel = None
ageModel = None
faceCounter = 0
frameCounter = -1
faceObjects = []


# ===================================================================
# main
# ===================================================================


def main():
    
    global faceModel
    global ageModel
    
    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)
    
    # face detection model
    faceModel = cv2.dnn.readNet('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')

    # age prediction model
    ageModel = tf.lite.Interpreter(model_path='age_from_face_model.tflite')  
    ageModel.allocate_tensors()
 
    st.title("Age Recognition Webcam")

    webrtc_ctx = webrtc_streamer(
        key="age-recognition-webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{'urls': ['stun:stun1.l.google.com:19302']}]},
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


# ===================================================================
# download models
# ===================================================================


# This file downloader demonstrates Streamlit animation.
# Used a code from https://github.com/streamlit/demo-self-driving/blob/master/streamlit_app.py
#
def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

 
# ===================================================================
# find all faces in image
# ===================================================================


def DetectFaces(net, frame_bgr, 
                conf_threshold=0.7, fit_square=True, fit_margin=1.4):

    # image height and width
    frameHeight = frame_bgr.shape[0]
    frameWidth = frame_bgr.shape[1]

    # transfer image to binary pixel object
    blob = cv2.dnn.blobFromImage(
        frame_bgr,  # image
        1.0,  # scalefactor
        (300, 300),  # size
        [104, 117, 123],  # values subtracted from channels
        True,  # swapRB
        False,
    )  # crop

    # set binary object as NN input
    net.setInput(blob)

    # faces detection
    detections = net.forward()

    # face rectangle borders
    faceBoxes = []

    # cycle through all rectangles found
    for i in range(detections.shape[2]):
        # get confidence if rectangle contains face
        confidence = detections[0, 0, i, 2]
        # compare with threshold, if exceeds - face detected
        if confidence > conf_threshold:
            # calculate rectangle coordinates
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            # get square across face
            if fit_square:

                x_mid = (x1 + x2) / 2
                y_mid = (y1 + y2) / 2
                size = max(x2 - x1, y2 - y1)

                # corners of face square
                x1 = int(x_mid - (size * fit_margin / 2))
                x2 = int(x_mid + (size * fit_margin / 2))
                y1 = int(y_mid - (size * fit_margin / 2))
                y2 = int(y_mid + (size * fit_margin / 2))

                # clamp coordinates to frame size
                x1 = max(0, min(x1, frameWidth))
                x2 = max(0, min(x2, frameWidth))
                y1 = max(0, min(y1, frameHeight))
                y2 = max(0, min(y2, frameHeight))

            faceBoxes.append([x1, y1, x2, y2])

    return faceBoxes


# ===================================================================
# crop images from input image
# ===================================================================


def CropImages(image, rects, resize, cvtColor):
    cropped_images = []
    for x1, y1, x2, y2 in rects:
        try:
            crop = image[y1:y2, x1:x2]
            if resize is not None:
                crop = cv2.resize(crop, resize)
            if cvtColor is not None:
                crop = cv2.cvtColor(crop, cvtColor)
            cropped_images.append(crop)
        except:
            print(f"Error crop image x1:{x1}, y1:{y1}, x2:{x2}, y2:{y2}")

    return cropped_images


# ===================================================================
# draw face rectangles
# ===================================================================


def HighlightFaces(image, faceObjects):

    # copy image to draw highlights
    _img = image.copy()

    # image height and width
    frameHeight = _img.shape[0]

    # draw rects
    for i, face in enumerate(faceObjects):

        if face.is_lost:
            color = (125, 125, 125)
        else:
            color = (0, 255, 0)

        x1, y1, x2, y2 = face.box

        # draw rectangle
        cv2.rectangle(_img, (x1,y1), (x2,y2), color, 
                      int(round(frameHeight/300)), 8)

        # draw value background
        cv2.rectangle(_img, (x1, y2 - 20), (x1 + 70, y2), color, -1, 8)

        # print age
        cv2.putText(
            _img,
            str(f"Age: {round(face.age)}"),
            org=(x1 + 2, y2 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # print ID
        cv2.putText(
            _img,
            str(f"ID: {face.id}"),
            org=(x1, y1 - 5),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return _img


# ===================================================================
# FaceObject class
# ===================================================================


class FaceObject:
    def __init__(self, id, box, age):

        self.id = id
        self.box = box
        self.age = age
        self.is_lost = False
        self.lost_frame = 0

    def set_age(self, age):
        if self.age == 0:
            self.age = age
        else:
            diff = age - self.age
            self.age += diff / 10


# ===================================================================
# pair together known and detected face coordinates.
# deactivate lost faces and put new faces in known list
# ===================================================================


def UpdateObjects(faceObjects=[], faceBoxes=[]):

    global faceCounter
    global frameCounter

    # get pairs of known and detected faces by center coordinates
    def get_pairs(list1, list2):
        # convert input to pandas dataframes
        pointsA = pd.DataFrame(list1, columns=["ID_A", "Lat_A", "Lng_A"])
        pointsA["key"] = 1
        pointsB = pd.DataFrame(list2, columns=["ID_B", "Lat_B", "Lng_B"])
        pointsB["key"] = 1

        # cross join known and detected centroids
        comparepoints = pd.merge(pointsA, pointsB, on="key")[
            ["ID_A", "Lat_A", "Lng_A", "ID_B", "Lat_B", "Lng_B"]
        ]

        # calculate distance between all points
        comparepoints["distance"] = (
            (comparepoints["Lat_A"] - comparepoints["Lat_B"]) ** 2
            + (comparepoints["Lng_A"] - comparepoints["Lng_B"]) ** 2
        ) ** (1 / 2)

        # aggregate for min distance and merge paired IDs
        comparepoints = pd.merge(
            comparepoints[["ID_A", "distance"]].groupby(["ID_A"]).min().reset_index(),
            comparepoints[["ID_B", "distance"]],
            on="distance",
        )

        comparepoints = comparepoints[["ID_A", "ID_B", "distance"]]
        comparepoints = comparepoints.sort_values("distance")
        comparepoints["uniq_pair"] = 0

        # find paired points. each point can be only in one pair
        used_A = []
        used_B = []
        for row in range(comparepoints.shape[0]):
            A = comparepoints.iloc[row, 0]
            B = comparepoints.iloc[row, 1]
            if A not in used_A and B not in used_B:
                # remember pair
                used_A.append(A)
                used_B.append(B)
                # mark pair as unique
                comparepoints.iloc[row, 3] = 1

        return used_A, used_B

    # get centroids of known objects
    known = []
    for i, face in enumerate(faceObjects):
        x_center = (face.box[0] + face.box[2]) / 2
        y_center = (face.box[1] + face.box[3]) / 2
        known.append([i, x_center, y_center])

    # get centroids of detected objects
    detected = []
    for i, rect in enumerate(faceBoxes):
        x_center = (rect[0] + rect[2]) / 2
        y_center = (rect[1] + rect[3]) / 2
        detected.append([i, x_center, y_center])

    # get pairs of known and detected faces
    matched_object_ids, matched_rect_ids = get_pairs(known, detected)

    # update position of known faces
    for pair in list(zip(matched_object_ids, matched_rect_ids)):
        faceObjects[pair[0]].box = faceBoxes[pair[1]]
        faceObjects[pair[0]].is_lost = False

    # deactivate and delete all lost faces
    for i in reversed(range(len(faceObjects))):

        # mark faces as lost
        if i not in matched_object_ids and not faceObjects[i].is_lost:
            faceObjects[i].is_lost = True
            faceObjects[i].lost_frame = frameCounter

        # delete obsolete faces
        if (
            faceObjects[i].is_lost
            and (frameCounter - faceObjects[i].lost_frame) > 20
        ):
            del faceObjects[i]

    # create of new detected faces
    for i in range(len(faceBoxes)):
        if i not in matched_rect_ids:
            new = FaceObject(faceCounter, faceBoxes[i], 0)
            faceCounter += 1
            faceObjects.append(new)

    return faceObjects


# ===================================================================
# frame callback
# ===================================================================


def video_frame_callback(frame_: av.VideoFrame) -> av.VideoFrame:

    # set up variables
    global frameCounter
    global faceModel
    global ageModel
    
    frame = frame_.to_ndarray(format="bgr24")
    refreshDetection = 1  # number of frames to next face detection
    conf_threshold = 0.5  # threshold of face detection algorithm
    squared = True  # use squared face boxes
    margin = 1.5  # faceBox margin multiplier
    refreshAge = 10  # number of frames to next age prediction

    # count this frame
    frameCounter += 1

    # detect faces
    if frameCounter % refreshDetection == 0:
        faceBoxes = DetectFaces(
            faceModel,
            frame,
            conf_threshold=conf_threshold,
            fit_square=squared,
            fit_margin=margin,
        )

        UpdateObjects(faceObjects, faceBoxes)

    # predict age
    if len(faceObjects) > 0 and frameCounter % refreshAge == 0:

        # get face boxes in correct order
        faceBoxes = []
        for i, face in enumerate(faceObjects):
            faceBoxes.append(face.box)

        # crop individual face images
        predImages_rgb = CropImages(
            frame, faceBoxes, resize=(224, 224), cvtColor=cv2.COLOR_BGR2RGB
        )

        # convert cropped images to model input
        predImages_rgb = [x / 255.0 for x in predImages_rgb]
        predImages_rgb = np.array(predImages_rgb, dtype=np.float32)

        # get predictions from tflite model
        preds = []
        input_details = ageModel.get_input_details()
        output_details = ageModel.get_output_details()
        for img in predImages_rgb:
            ageModel.set_tensor(input_details[0]["index"], [img])
            ageModel.invoke()
            preds.append(ageModel.get_tensor(output_details[0]["index"]))

        # put predicted age to faceObjects
        for i, face in enumerate(faceObjects):
            face.set_age(int(preds[i]))

    # highlight faces on frame image
    frame = HighlightFaces(frame, faceObjects)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")


if __name__ == "__main__":
    main()     
