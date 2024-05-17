import cv2
from ultralytics import YOLO


yolo_model = YOLO(r"C:\Users\Rahsh\OneDrive\Desktop\Rahaf_Fatma_Lama_Project\Rahaf_Fatma_Lama_Project\Project_Files\Detect_Vehicles\model.pt")


vehicle_groups = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']


class_colors = {
    'car': (0, 255, 0),         # green
    'truck': (0, 0, 255),       # red
    'bus': (255, 0, 0),         # blue
    'motorcycle': (0, 255, 255),# yellow
    'bicycle': (128, 0, 128)    # purple
}

video_path = r"C:\Users\Rahsh\OneDrive\Desktop\Rahaf_Fatma_Lama_Project\Rahaf_Fatma_Lama_Project\Project_Files\Detect_Vehicles\Tested_Videos\vehicles_test4.MOV"
video = cv2.VideoCapture(video_path)  

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

output_video_path = r"C:\Users\Rahsh\OneDrive\Desktop\Rahaf_Fatma_Lama_Project\Rahaf_Fatma_Lama_Project\Project_Files\Detect_Vehicles\Output_of_Detected_Vehicles\output_video4.mp4"
new_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:
    ret, frame = video.read()

    if not ret:
        break

    results = yolo_model(frame)

    for result in results:
        boxes = result.boxes  
        classes = result.boxes.cls  

        for box, cls in zip(boxes.xyxy.tolist(), classes.tolist()):
            class_name = yolo_model.names[int(cls)]

            if class_name in vehicle_groups:
                color = class_colors.get(class_name, (255, 255, 255))  

                x1, y1, x2, y2 = [int(val) for val in box]

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 2)

    new_video.write(frame)

    cv2.imshow('Object Detection for Vehicles', frame)

    if cv2.waitKey(7) & 0xFF == ord('a'):
        print("Video playback interrupted by user!")
        break

video.release()
new_video.release()
cv2.destroyAllWindows()

print("Output video saved as", output_video_path)
