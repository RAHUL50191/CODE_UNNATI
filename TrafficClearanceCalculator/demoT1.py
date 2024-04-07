import cv2
import pandas as pd
import numpy as np
# from AccidentDetection.detection import AccidentDetectionModel
from ultralytics import YOLO
from ultralytics.solutions import object_counter

# Amodel = AccidentDetectionModel("AccidentDetection/model.json", 'AccidentDetection/model_weights.keras')
font = cv2.FONT_HERSHEY_SIMPLEX
model = YOLO("yolov8n.pt")
caps = ["lane1.webm", "lane2.mp4", "lane3.mp4", "lane4.webm"]

x1 = 1200
y1 = 1024

def getRegionPoints(x1, y1):
    div_x1 = x1 // 2
    div_y1 = y1 // 2

    quarter_width = x1 - div_x1
    quarter_height = y1 - div_y1

    part_width = quarter_width // 3
    part_height = quarter_height // 3

    return [
        [(div_x1, div_y1), (x1, div_y1), (x1, div_y1 + part_height), (div_x1, div_y1 + part_height)],
        [(div_x1, div_y1 + part_height), (x1, div_y1 + part_height), (x1, div_y1 + 2 * part_height), (div_x1, div_y1 + 2 * part_height)],
        [(div_x1, div_y1 + 2 * part_height), (x1, div_y1 + 2 * part_height), (x1, y1), (div_x1, y1)]
    ]

region_points = getRegionPoints(x1, y1)
classes_to_count = [2, 5, 7]  # car,truck,bus

def filter_dict_by_keys(input_dict, keys_list):
    return {key: input_dict[key] for key in keys_list}

def initialize_counters(region_points):
    counters = []
    for points in region_points:
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=False, reg_pts=points, classes_names=filter_dict_by_keys(model.names, classes_to_count), draw_tracks=True)
        counters.append(counter)
    return counters

def Process(i, timeout=50):
    cap = cv2.VideoCapture(caps[i])
    counters = initialize_counters(region_points)

    while cap.isOpened():
        timer = 0
        f = True
        start_time = cv2.getTickCount()

        while True:
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            frame = cv2.resize(frame, (x1, y1))
            tracks = model.track(frame, persist=True, show=False, classes=classes_to_count, tracker="bytetrack.yaml")

            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # roi = cv2.resize(gray_frame, (250, 250))
            
            # pred, prob = Amodel.predict_accident(roi[np.newaxis, :, :])
            # if pred == "Accident" and prob.all() > 0.9:
            #     print("beep")

            # cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            # cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

            if f:
               for index, counter in enumerate(counters):
                    tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)
                    frame = counter.start_counting(frame, tracks)
                    px = pd.DataFrame(tracks[0].boxes.data.detach().cpu().numpy()).astype("float")
                    counts1 = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}

                    for _, row in px.iterrows():
                        class_num = row[6]
                        if class_num in classes_to_count:
                            counts1[class_num] += 1
                
                    dist = index*2
                    timer += sum([
                        9.5 + dist if counts1.get(7, 0) != 0 else 
                        7.5 + dist if counts1.get(5, 0) != 0 else 
                        8.5 + dist if counts1.get(3, 0) != 0 else 
                        6.5 + dist if counts1.get(2, 0) != 0 else 0
                    ])
                    
                    f = False

            print("Calculated time for Road ", i, "is :", timer)
            # for k in region_points:
                # cv2.polylines(frame, [np.array(k)], isClosed=True, color=(255, 0, 0), thickness=2)

            cv2.imshow(f"frame{i}", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()

            if elapsed_time >= timer or elapsed_time >= timeout:
                break

        cap.release()
        if elapsed_time >= timer or elapsed_time >= timeout or timer >= timeout:
            break

def main():
    for i in range(4):
        Process(i)

if __name__ == "__main__":
    main()
                          
