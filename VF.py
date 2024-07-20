# -*- coding: utf-8 -*-
"""
Created on Fri May 31 15:10:52 2024

@author: sribr
"""

import cv2

# Open the video file
video_path = 'input_video.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize frame count
frame_count = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was not read successfully, break the loop
    if not ret:
        break

    # Display the frame
    cv2.imshow('Frame', frame)

    # Save the frame as an image (optional)
    cv2.imwrite(f'frame_{frame_count}.jpg', frame)  # Save frame as an image file

    # Increment frame count
    frame_count += 1

    # Wait for 'q' key press to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
