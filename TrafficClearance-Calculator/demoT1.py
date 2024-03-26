import cv2
import asyncio
import pandas as pd
from ultralytics import YOLO
from ultralytics.solutions import object_counter

model = YOLO("yolov8n.pt")
caps = [(f"lane{1}.webm"),(f"lane{2}.mp4"),(f"lane{3}.mp4"),(f"lane{4}.webm") ]
x1 = 1200
y1 = 1024
region_points = [
    [(x1 / 2, y1 / 2), (x1 - 20, y1 / 2), (x1 - 20, y1 - 20), (x1 / 2, y1 - 20)],
    [(x1 - 20, y1 / 2), (x1 - 20, y1 / 2), (x1 - 20, y1 - 20), (x1 - 20, y1 - 20)],
    [(20, y1 / 2), (x1 - 20, y1 / 2), (x1 - 20, y1 - 20), (20, y1 - 20)]
]

# Classes to count
classes_to_count = [1, 2, 5, 7]

def filter_dict_by_keys(input_dict, keys_list):
    return {key: input_dict[key] for key in keys_list}

def initialize_counters(region_points):
    counters = []
    for points in region_points:
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=False, reg_pts=points, classes_names=filter_dict_by_keys(model.names, classes_to_count), draw_tracks=True)
        counters.append(counter)
    return counters

async def Process(i,timeout=50):
    cap = cv2.VideoCapture(caps[i])

    # Initialize Object Counters for each lane
    counters = initialize_counters(region_points)

    while True:
        timer = 0
        f=True
        start_time = asyncio.get_event_loop().time()  # Start time of the loop

        while cap.isOpened() and timer < timeout:
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break

            frame = cv2.resize(frame, (x1, y1))
            tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)

         
            if f:
                for counter in counters:
                    tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)
                    frame = counter.start_counting(frame, tracks)
                    px = pd.DataFrame(tracks[0].boxes.data.detach().cpu().numpy()).astype("float")
                    counts1 = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}
                    # counts2 = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}
                    # counts3 = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 7: 0}
                    for _, row in px.iterrows():
                        class_num = row[6]
                        if class_num in classes_to_count:
                            counts1[class_num] += 1
                
                    # for _, row in px.iterrows():
                    #     class_num = row[6]
                    #     if class_num in classes_to_count:
                    #         counts2[class_num] += 1
                    
                    # for _, row in px.iterrows():
                    #     class_num = row[6]
                    #     if class_num in classes_to_count:
                    #         counts3[class_num] += 1
                    dist=0
                    timer = sum([
                        7 + dist if counts1.get(2, 0) != 0 else 0,
                        5 + dist if counts1.get(3, 0) != 0 else 0,
                        10 + dist if counts1.get(5, 0) != 0 else 0,
                        10 + dist if counts1.get(7, 0) != 0 else 0
                    ])
                    print("Calculated time for Road ",i,"is :",timer)
                    f=False
            
            #display video
            if success == True: 
                cv2.imshow((f"frame{i}"),frame) 
          
            # Press Q on keyboard to exit 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break
   
            # await asyncio.sleep(timer)
            # Check if timeout has occurred
            if asyncio.get_event_loop().time() - start_time >= timer or asyncio.get_event_loop().time() - start_time >=timeout:
                break
        cap.release()   
        if asyncio.get_event_loop().time() - start_time >= timer or asyncio.get_event_loop().time() - start_time >=timeout or timer >= timeout:  
            break

async def main():
    functions = [Process(0), Process(1), Process(2), Process(3)]

    for func in functions:
        try:
            # Call the function and get the sleep time
            await asyncio.wait_for(func, timeout=40)
        except asyncio.TimeoutError:
            print("Timeout occurred for the function")

if __name__ == "__main__":
    asyncio.run(main())