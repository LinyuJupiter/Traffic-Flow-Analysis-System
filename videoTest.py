import cv2
import imageio
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('car_clp_1.pt')

# Open the video file
video_path = "001.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
# 创建自定义窗口
cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
# 调整窗口大小以适应屏幕
cv2.resizeWindow("YOLOv8 Inference", 160 * 8, 90 * 8)  # 根据需要设置窗口大小
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
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        gif_images.append(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
imageio.mimsave('animated.gif', gif_images, duration=40)
