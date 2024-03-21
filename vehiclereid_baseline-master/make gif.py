import cv2
from ultralytics import YOLO
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models import *
import numpy as np
import imageio

# Load the YOLOv8 model
# gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = cnn.CNN(num_classes=57).to(device)
state_dict = torch.load(r"./checkpoints/att/Car_epoch_15.pth")
net.load_state_dict(state_dict['state_dict'])
net.eval()
Vehicle_model = ["SUV", "car", "SUV", "van", "car", "van", "van", "car", "goods van", "SUV", "van", "car", "van", "bus",
                 "car", "truck", "car", "car", "SUV", "van", "van", "SUV", "car", "truck", "SUV", "SUV", "bus", "SUV",
                 "car", "ambulance", "car", "Large truck", "Large truck", "SUV", "van", "car", "goods van", "goods van",
                 "Large truck", "Large truck", "Large truck", "Large truck", "Tank truck", "Large truck", "Large truck",
                 "Tank truck",
                 "Tank truck", "car", "truck", "Large truck", "car", "Large truck", "truck", "car", "car", "goods van",
                 "van"]


def load_and_preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Resize((100, 100)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图片归一化
                             std=[0.229, 0.224, 0.225])
    ])
    preprocessed_image = preprocess(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)  # 添加额外的批次维度
    return preprocessed_image


model = YOLO(r'../car_clp_1.pt')

# Open the video file
video_path = r'../001.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
# 创建自定义窗口
# cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
# 调整窗口大小以适应屏幕
# cv2.resizeWindow("YOLOv8 Inference", 160 * 8, 90 * 8)  # 根据需要设置窗口大小

# 初始化车辆计数器
vehicle_count = 0
vehicle_up = 0
vehicle_down = 0
# 创建一个空的字典，用于跟踪每辆车的进入和离开
vehicles = {}
gif_images = []
times = 0
while cap.isOpened():
    times += 1
    print(times, "/3000", sep="")
    if times == 200:
        break
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 inference on the frame
        results = model.track(frame, persist=True)
        # print(results)
        # 获取检测到的车辆边界框
        bboxes = results[0].boxes
        frame_copy = frame
        # 添加标识线
        indices = np.indices(frame.shape[:2])
        # 计算 x 和 y 的坐标值
        x = indices[0]
        y = indices[1]
        multiple = 6
        # 创建一个满足条件的布尔掩码
        mask = (multiple * x - y) == int((frame.shape[0] / 3) * multiple)
        # 将满足条件的元素设置为 0
        frame[mask] = 0

        # 遍历每个边界框
        for bbox in bboxes:
            # 提取边界框的坐标
            x1, y1, x2, y2 = bbox.xyxy.tolist()[0]
            position = [(x2 + x1) / 2, (y2 + y1) / 2]  # 车辆中心点坐标
            # 构造车辆ID
            vehicle_id = str(int(bbox.id.tolist()[0]))
            # 如果是新的车辆
            if vehicle_id not in vehicles:
                # 添加车辆到字典中
                vehicles[vehicle_id] = {"position": position, "UPorNOT": False}
            # 检测up还是down
            if position[1] < vehicles[vehicle_id]["position"][1]:
                vehicles[vehicle_id]["UPorNOT"] = True
            # 过线检测
            if multiple * position[1] - position[0] >= int((frame.shape[0] / 3) * multiple) >= multiple * \
                    vehicles[vehicle_id]["position"][1] - vehicles[vehicle_id]["position"][0]:
                vehicle_down += 1
                vehicle_count += 1
                vehicles[vehicle_id]["position"] = position
            elif multiple * position[1] - position[0] <= int((frame.shape[0] / 3) * multiple) <= multiple * \
                    vehicles[vehicle_id]["position"][1] - vehicles[vehicle_id]["position"][0]:
                vehicle_up += 1
                vehicle_count += 1
                vehicles[vehicle_id]["position"] = position
            else:
                pass
            ff = frame_copy[int(y1):int(y2), max(int(x1), 0):max(int(x2), 0), :]
            img = load_and_preprocess_image(ff)
            with torch.no_grad():
                outputs = net(img)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 应用 softmax 函数获得概率
            index = torch.argmax(probabilities[0]).tolist()
            probability = probabilities[0][index].tolist()
            if vehicles[vehicle_id]["UPorNOT"]:
                describe = "ID:" + vehicle_id + " " + Vehicle_model[index] + "  Conf:" + str("%.2f" % probability)
            else:
                describe = "ID:" + vehicle_id + "  Conf:" + str("%.2f" % probability)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, describe, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # frame[int(frame.shape[0] // 3), :, :] = 0  # 添加识别线

        cv2.putText(frame, "vehicle_count:" + str(vehicle_count), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        cv2.putText(frame, "vehicle_up:" + str(vehicle_up), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        cv2.putText(frame, "vehicle_down:" + str(vehicle_down), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 0, 0), 2)
        # Display the annotated frame
        # cv2.imshow("YOLOv8 Inference", frame)
        gif_images.append(frame)
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
imageio.mimsave('animated.gif', gif_images, duration=40)
