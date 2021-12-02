"""
Project: Computer vision demo
Module:  Main
Author:  Andrea Gonzalez Silva
Version: 1.0
"""

# Import modules
from packages.UI.video_handler import VideoHandler

if __name__ == '__main__':

    # Define the video source and image processing type
    video_source = 0
    processing_type = "detect_face"

    # Start the streaming video and image processing
    video_handler = VideoHandler()
    video_handler.stream_video(video_source, processing_type)
