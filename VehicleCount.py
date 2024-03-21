import cv2
from ultralytics import YOLO

"""
根据具体的车道，修改识别线（第50/54/63行的代码）
默认识别线为图片中间
"""
# Load the YOLOv8 model
model = YOLO('car_clp_1.pt')

# Open the video file
video_path = "002.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
# 创建自定义窗口
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
# 调整窗口大小以适应屏幕
cv2.resizeWindow("YOLOv8 Inference", 160 * 8, 90 * 8)  # 根据需要设置窗口大小

# 初始化车辆计数器
vehicle_count = 0
vehicle_up = 0
vehicle_down = 0
# 创建一个空的字典，用于跟踪每辆车的进入和离开
vehicles = {}
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True)
        # print(results)
        # 获取检测到的车辆边界框
        bboxes = results[0].boxes

        # 遍历每个边界框
        for bbox in bboxes:
            # 提取边界框的坐标
            x1, y1, x2, y2 = bbox.xyxy.tolist()[0]
            position = [(x2 + x1) / 2, (y2 + y1) / 2]
            # 构造车辆ID
            vehicle_id = str(bbox.id)

            # 如果是新的车辆
            if vehicle_id not in vehicles:
                # 添加车辆到字典中
                vehicles[vehicle_id] = {"position": position}
            if position[1] >= frame.shape[0] // 2 >= vehicles[vehicle_id]["position"][1]:
                vehicle_down += 1
                vehicle_count += 1
                vehicles[vehicle_id]["position"] = position
            elif position[1] <= frame.shape[0] // 2 <= vehicles[vehicle_id]["position"][1]:
                vehicle_up += 1
                vehicle_count += 1
                vehicles[vehicle_id]["position"] = position
            else:
                pass

                # Visualize the results on the frame
        annotated_frame = results[0].plot()
        annotated_frame[int(frame.shape[0] // 2), :, :] = 0  # 添加识别线
        cv2.putText(annotated_frame, "vehicle_count:" + str(vehicle_count), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        cv2.putText(annotated_frame, "vehicle_up:" + str(vehicle_up), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        cv2.putText(annotated_frame, "vehicle_down:" + str(vehicle_down), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
print("vehicle_count:" + str(vehicle_count))
print("vehicle_up:" + str(vehicle_up))
print("vehicle_down:" + str(vehicle_down))
