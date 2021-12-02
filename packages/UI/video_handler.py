"""
Project: Computer vision demo
Module:  Video Handler
Author:  Andrea Gonzalez Silva
Version: 1.0
"""

# Import modules
import cv2
from packages.image_processing.image_processing import ImageProcessing


class VideoHandler:

    @staticmethod
    def stream_video(video_source, processing_type=None):
        """
        Streams the video frame of the specified video camera source
        :param video_source: The video camera source
        :param processing_type: Process type to include in the frame: "detect_face"
        """

        # Initialise the video capture module
        video = cv2.VideoCapture(video_source)
        image_processing = ImageProcessing()

        # The video streaming loop, it will continue until user input indicating program to exit
        while True:

            # Retrieve the video frame
            ret, frame = video.read()

            if ret:
                # Process the frame and add the identified face ROI
                processed_frame = image_processing.process_frame(frame=frame, processing_type=processing_type)

                # Display the processed frame
                cv2.imshow('demo', processed_frame)

            # Check if a user input has been received and if it is the 'q' key, then quit the program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the user indicated to quit the program, release the video and destroy
        video.release()
        cv2.destroyAllWindows()
