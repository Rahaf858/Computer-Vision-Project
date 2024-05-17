This project consists of two main parts: a handwritten digit recognition application using a pre-trained neural network and a vehicle detection application using YOLOv8.

Part 1: Handwritten Digit Recognition
The first part of the project is a web application built with Streamlit that allows users to draw digits on a canvas. The application uses a pre-trained Convolutional Neural Network (CNN) to recognize and predict the drawn digit.

Features:

User can draw digits on a 300x300 canvas.
The drawing is preprocessed and resized to 28x28 pixels, then passed to the CNN model for prediction.
The predicted digit is displayed on the screen along with the drawn image.
Technologies Used:

Streamlit for the web interface.
OpenCV for image processing.
TensorFlow/Keras for loading the pre-trained model.
Usage:

Clone the repository.
Install the required packages: pip install streamlit opencv-python-headless tensorflow streamlit-drawable-canvas.
Run the Streamlit app: streamlit run app.py.


Part 2: Vehicle Detection
The second part of the project involves detecting different types of vehicles in a video using the YOLOv8 model.

Features:

Detects and classifies vehicles such as cars, trucks, buses, motorcycles, and bicycles.
Draws bounding boxes around detected vehicles with class-specific colors.
Outputs a video with the detected and labeled vehicles.

Technologies Used:

OpenCV for video processing.
YOLOv8 model for object detection.
Usage:

Clone the repository.
Install the required packages: pip install opencv-python ultralytics.
Ensure you have the YOLOv8 model file (yolov8n.pt) in the appropriate directory.
Update the video path in the script to your input video.
Run the detection script: python detect_vehicles.py.
