import cv2
from ultralytics import YOLO
import os

# Initialize a YOLO-World model
model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# Define custom classes
# model.set_classes(["person"])

current_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(current_dir, "../dataset/12910250-hd_1920_1080_30fps.mp4")
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 

    # Execute prediction on the frame
    results = model.predict(frame, conf=0.40, classes=0)

    bboxes = []
    confs = []
    class_ids = []

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs

        xyxy = boxes.xyxy.cpu().numpy()

        if xyxy.size == 0:
            continue

        conf = boxes.conf.cpu().numpy()
        class_id = boxes.cls.cpu().numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            bboxes.append([x1, y1, x2, y2])
            confs.append(conf[i])
            class_ids.append(class_id[i])


    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]

        conf = confs[i]
        class_id = class_ids[i]
        fps = 0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.1f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        # cv2.putText(frame, 'FPS: {:.1f}'.format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Write the frame into the file 'output.mp4'
    out.write(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
