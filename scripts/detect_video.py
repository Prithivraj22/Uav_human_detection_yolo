import cv2
import numpy as np
import os

net = cv2.dnn.readNet("d:/Mini_project/models/yolov3.weights", "d:/Mini_project/uavhuman/models/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
with open("d:/Mini_project/models/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def detect_video(video_path, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Unable to open video writer for {output_path}")
        cap.release()
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 3 == 0:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            class_ids = []
            confidences = []
            boxes = []
            for detection_out in outs:
                for detection in detection_out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == "person":
  
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)


                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            video_writer.write(frame)
            cv2.imshow("Frame", frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    if hasattr(cv2, 'destroyAllWindows'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_video("d:/Mini_project/data/videos/RC_0004_2108030748.MP4", "d:/Mini_project/uavhuman/output/detected_videos/output_video.avi", playback_speed=2)
