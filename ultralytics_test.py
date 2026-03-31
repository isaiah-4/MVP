from ultralytics import YOLO

model = YOLO("Models/player_detector.pt")

results = model.track(source="Input_vids/video_1.mp4", save=True)
print(results)
print("(=====================================)")

for box in results[0].boxes:
    print(box)
    
