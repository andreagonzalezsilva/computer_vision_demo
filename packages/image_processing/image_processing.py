"""
Project: Computer vision demo
Module:  Video Handler
Author:  Andrea Gonzalez Silva
Version: 1.0
"""

# Import modules
import cv2
import os


class ImageProcessing:

    def __init__(self):
        pass

    def process_frame(self, frame=None, processing_type=None):
        """
        Process the frame depending on the specified processing type
        :param frame: Current streaming frame
        :param processing_type: Process type to include in the frame: "detect_face"
        :return: The frame with the highlighted ROI frame
        """

        # In case there is not specified process, return the input frame
        if not processing_type:
            return frame

        # In the case of 'detect_face' image processing type, add the detected faces in the current frame
        elif processing_type == "detect_face":
            processed_frame = self.__detect_facial_features(frame)

        # If there is an invalid image processing type
        else:
            raise NameError("Invalid frame process type. Please select a valid process type: \'detect_face\'.")

        #  Return the processed frame
        return processed_frame

    @staticmethod
    def __detect_facial_features(frame):
        """
        Detect the facial features which are the face and the eyes
        :param frame: Current streaming frame
        :return: The frame with the highlighted facial features
        """

        # Convert the frame to grayscale for face and eye detection
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Load the face and eye cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Detect the faces and store the faces coordinates and size using a multi scale method
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)

        # Add to the frame the rectangles highlighting to the frame for each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (153, 153, 0), 2)

            # Extract the face ROI in grayscale and colour
            region_gray = gray_image[y:y + h, x:x + w]
            region_colour = frame[y:y + h, x:x + w]

            # Detect the eyes and store the faces coordinates and size using a multi scale method
            eyes = eye_cascade.detectMultiScale(region_gray, 1.1,   3)

            # Add to the frame the rectangles highlighting for each detected face
            for (eye_x, eye_y, eye_w, eye_h) in eyes:
                cv2.rectangle(region_colour, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (24, 215, 245),  2)

        # Return the processed frame
        return frame

