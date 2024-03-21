import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os
import numpy as np
from tkinter import messagebox
import threading
from tkinter import ttk  # ttk包含了一些更现代的widget，如Combobox
import time

root = tk.Tk()
root.geometry("1000x600")
root.resizable(False, False)

# 创建两个StringVar对象来分别存储图像和视频的文件路径
file_path_var_image = tk.StringVar()
file_path_var_video = tk.StringVar()

# 创建一个用于显示图片的Canvas
# 刚开始我用的是label来显示，后来发现选中图片或视频后会改变大小来适应图片
# 从而改变了整个窗口的布局，所以使用Canvas来显示图片
canvas = tk.Canvas(root, width=800, height=450, bg='white', highlightbackground="yellow", highlightthickness=2)
canvas.place(x=5, y=130)

runnin = False


# 打开图片
def open_image_file():
    global running
    running = False
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    file_path_var_image.set(file_path)
    file_path_var_video.set('')  # 清空视频文件路径显示
    # 用PIL库打开图像并显示在Canvas上
    image = Image.open(file_path)
    image.thumbnail((800, 600))  # 把图像缩放到合适的大小以适应canvas
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.image = photo  # 保存对PhotoImage对象的引用，防止被垃圾回收

    text1.configure(state='normal')
    text1.delete(1.0, 'end')
    text1.insert(1.0, file_path_var_image.get())
    text1.configure(state='disabled')
    # 在选中图片后，要把选中视频的路径清除
    text2.configure(state='normal')  # 允许修改文本框内容
    text2.delete(1.0, 'end')  # 清除文本框内容
    text2.configure(state='disabled')  # 设置为只读模式


# 初始化车辆计数器
vehicle_count = 0
vehicle_up = 0
vehicle_down = 0
# 创建一个空的字典，用于跟踪每辆车的进入和离开
vehicles = {}


# 打开视频
def open_video_file():
    global running
    running = False
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi *.mp4")])
    file_path_var_video.set(file_path)
    file_path_var_image.set('')  # 清空图像文件路径显示
    # 选择视频第一帧作为输入图像
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)

    image.thumbnail((800, 600))  # 把图像缩放到合适的大小以适应canvas
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor='nw', image=photo)
    canvas.image = photo  # 保存对PhotoImage对象的引用，防止被垃圾回收
    cap.release()

    text2.configure(state='normal')
    text2.delete(1.0, 'end')
    text2.insert(1.0, file_path_var_video.get())
    text2.configure(state='disabled')

    # 在选中视频后，要把选中图片路径清除
    text1.configure(state='normal')  # 允许修改文本框内容
    text1.delete(1.0, 'end')  # 清除文本框内容
    text1.configure(state='disabled')  # 设置为只读模式


root.title("车辆检测系统")

# 在窗口的最上方添加一个标题
title = tk.Label(root, text="车辆检测系统", font=("Arial", 24, "bold"))
# title.grid(row=0, column=0, columnspan=2)  # 跨越两列
title.place(x=300, y=5)

button1 = tk.Button(root, text="打开图片📸", command=open_image_file, bg='blue', fg='white', borderwidth=2,
                    relief='raised')
button1.place(x=35, y=45)

text1 = tk.Text(root, height=1, width=55, bg='light gray', font=('Arial', 12))
text1.place(x=110, y=50)

button2 = tk.Button(root, text="打开视频📹", command=open_video_file, bg='green', fg='white', borderwidth=2,
                    relief='raised')
button2.place(x=35, y=80)

text2 = tk.Text(root, height=1, width=55, bg='light gray', font=('Arial', 12))
text2.place(x=110, y=90)

# Load the YOLOv8 model
model = YOLO('./car_clp_1.pt')


def detect(img_cv):
    # 这里是用来检测图像的代码
    results = model(img_cv)  # 进行识别
    result = results[0]
    # print(result.boxes)
    # print(result.probs)
    img_with_boxes = result.plot(conf=True, boxes=True, labels=True)  # 画框框
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)  # 改格式

    detection = [
        {
            "class": result.names[int(c)],
            "bounding_box": box,
            "confidence": conf
        }
        for c, box, conf in zip(result.boxes.cls, result.boxes.xyxy, result.boxes.conf)
    ]

    return img_with_boxes_rgb, detection


def detect_image(image_path):
    # 首先检查传入的参数是否为字符串类型
    if isinstance(image_path, str):
        img_cv = cv2.imread(image_path)  # 如果是字符串，那么就认为这是一个文件路径，使用cv2.imread()读取图片
    else:
        # 如果不是字符串，那么就认为这是一个PIL Image对象，使用numpy.array()将其转换为numpy数组
        # 然后将颜色空间从RGB转换为BGR，因为OpenCV的颜色空间是BGR，而PIL Image的颜色空间是RGB
        img_cv = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)

    img_with_boxes_rgb, detection = detect(img_cv)

    img_with_boxes_pil = Image.fromarray(img_with_boxes_rgb)  # 改成PIL图片，因为要在GUI中显示

    # return img_with_boxes_pil, detection
    return img_with_boxes_pil, detection  # ok，现在返回


def detect_video(video_path):
    global vehicle_count, vehicle_up, vehicle_down, vehicles
    vehicle_count = 0
    vehicle_up = 0
    vehicle_down = 0
    vehicles = {}

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 创建一个VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    global running
    running = True  # 加这个running是为了能够看视频看一半选择新的图片或视频时能够暂停

    global playing

    parameters_updated = False
    # while running:
    while cap.isOpened():
        while running:
            if playing:

                # 读取下一帧
                ret, frame = cap.read()
                if not ret:
                    break  # 如果没有更多的帧，就退出循环

                # 对帧应用车辆检测
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                detected_img, detection = detect(pil_img)
                frame = cv2.cvtColor(np.array(detected_img), cv2.COLOR_RGB2BGR)

                # 检测车辆计数
                # Run YOLOv8 inference on the frame
                results = model.track(frame, persist=True)

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

                annotated_frame = results[0].plot()
                annotated_frame[int(frame.shape[0] // 2), :, :] = 0  # 添加识别线
                cv2.putText(annotated_frame, "vehicle_count:" + str(vehicle_count), (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0), 2)
                cv2.putText(annotated_frame, "vehicle_up:" + str(vehicle_up), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (255, 0, 0), 2)
                cv2.putText(annotated_frame, "vehicle_down:" + str(vehicle_down), (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (255, 0, 0), 2)

                # Convert the image from BGR color (which OpenCV uses) to RGB color
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Convert the Image object into a TkPhoto object
                im = Image.fromarray(img)
                im.thumbnail((800, 600))  # 缩放图像以适应canvas
                imgtk = ImageTk.PhotoImage(image=im)

                # Clear previous items on the canvas
                canvas.delete('all')

                # Put the image on the canvas by creating an image object, and then drawing it
                canvas.create_image(0, 0, anchor='nw', image=imgtk)
                canvas.image = imgtk  # 保存对PhotoImage对象的引用，防止被垃圾回收

                # Add center line
                canvas.create_line(0, frame.shape[0] // 2, frame.shape[1], frame.shape[0] // 2, fill='red')

                # Add texts
                canvas.create_text(30, 40, fill="blue", font="Times 20 italic bold",
                                   text="Vehicle count: " + str(vehicle_count))
                canvas.create_text(30, 80, fill="blue", font="Times 20 italic bold",
                                   text="Vehicle up: " + str(vehicle_up))
                canvas.create_text(30, 120, fill="blue", font="Times 20 italic bold",
                                   text="Vehicle down: " + str(vehicle_down))

                # Put the image on   the canvas by creating an image object, and then drawing it
                canvas.create_image(0, 0, anchor='nw', image=imgtk)
                canvas.image = imgtk  # 保存对PhotoImage对象的引用，防止被垃圾回收

                # 把检测到的车辆标注出来，并写入新的帧
                out.write(frame)

                # 在视频播放时重置变量
                parameters_updated = False

                # 等待GUI更新
                root.update_idletasks()
                root.update()
            else:
                # 只有在参数还没有被更新时才更新它们
                if not parameters_updated:
                    update_right_part(detection)
                    parameters_updated = True
                time.sleep(0.1)

    # 关闭视频文件
    # cap.release()
    out.release()


playing = True


def play_pause():  # 抓拍功能，相当于暂停
    global playing
    playing = not playing
    button4.config(text="抓拍" if playing else "继续")


button4 = tk.Button(root, text="抓拍/继续", command=play_pause, font=("Arial", 14), bg="darkblue", fg="white")
button4.place(x=850, y=450)

right_frame = tk.Frame(root, width=300, height=350)
right_frame.place(x=810, y=50)

label_num_objects = tk.Label(right_frame, text="检测对象数量:", font=("Arial", 10), bg="white", fg="black")
label_num_objects.place(x=10, y=0)
text_num_objects = tk.Text(right_frame, height=1, width=5)
text_num_objects.place(x=103, y=3)

# 创建一个Combobox来展示所有检测到的对象

label_select = tk.Label(right_frame, text="在检测后选择对象进行查看：", font=("Arial", 10), bg="white", fg="black")
label_select.place(x=10, y=40)
combo_var = tk.StringVar()
combo = ttk.Combobox(right_frame, textvariable=combo_var, width=20)
combo.place(x=10, y=63)

# 创建一个Text控件来展示所选对象的详细信息
label_box = tk.Label(right_frame, text="BOX:", font=("Arial", 10), bg="white", fg="black")
label_box.place(x=10, y=100)
text_box = tk.Text(right_frame, height=10, width=23)
text_box.place(x=10, y=120)

label_class = tk.Label(right_frame, text="class:", font=("Arial", 10), bg="white", fg="black")
label_class.place(x=10, y=260)
text_class = tk.Text(right_frame, height=1, width=10)
text_class.place(x=53, y=263)

label_confidence = tk.Label(right_frame, text="confidence:", font=("Arial", 10), bg="white", fg="black")
label_confidence.place(x=10, y=290)

text_conf = tk.Text(right_frame, height=1, width=10)
text_conf.place(x=83, y=293)


# 用于更新右边控件的函数
def update_right_part(detection):
    # 清空Combobox和Text控件
    combo['values'] = ()
    text_num_objects.delete(1.0, tk.END)
    text_box.delete(1.0, tk.END)
    text_class.delete(1.0, tk.END)
    text_conf.delete(1.0, tk.END)

    # 用所有检测到的对象更新Combobox
    combo['values'] = [f'Object {i + 1}' for i in range(len(detection))]
    # 更新检测到的对象的数量
    text_num_objects.insert(tk.END, str(len(detection)))

    # 创建一个函数来处理当Combobox的选项改变时的事件
    def on_combo_changed(event):
        # 获取当前选中的对象的索引
        index = combo.current()
        if index >= 0:  # 如果有对象被选中
            detect = detection[index]
            # 清空并更新Text控件的内容
            text_box.delete(1.0, tk.END)
            # print(str(detection["bounding_box"]))
            x1 = detection[index]["bounding_box"][0].item()
            y1 = detection[index]["bounding_box"][1].item()
            x2 = detection[index]["bounding_box"][2].item()
            y2 = detection[index]["bounding_box"][3].item()
            txt = 'x1: ' + str(x1) + '\n' + '\n' + 'y1: ' + str(y1) + '\n' + '\n' + 'x2: ' + str(
                x2) + '\n' + '\n' + 'y2: ' + str(y2) + '\n'

            text_box.insert(tk.END, txt)
            text_class.delete(1.0, tk.END)
            text_class.insert(tk.END, detect["class"])
            text_conf.delete(1.0, tk.END)
            text_conf.insert(tk.END, detect["confidence"].item())

    # 绑定这个函数到Combobox的<<ComboboxSelected>>事件上
    combo.bind('<<ComboboxSelected>>', on_combo_changed)


def start_detection():
    global playing

    # 获取图像和视频文件路径
    image_path = file_path_var_image.get()
    video_path = file_path_var_video.get()
    # 根据文件类型调用不同的处理代码
    if image_path:  # 如果有图像文件路径

        playing = False
        extension = os.path.splitext(image_path)[1].lower()
        if extension in ['.jpg', '.jpeg', '.png']:
            # 调用你的车辆检测代码
            image, detection = detect_image(image_path)

            # 并将结果显示到canvas中
            image.thumbnail((800, 600))  # 把图像缩放到合适的大小以适应canvas
            photo = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor='nw', image=photo)
            canvas.image = photo  # 保存对PhotoImage对象的引用，防止被垃圾回收

            # 更新右边的
            update_right_part(detection)
            pass
    elif video_path:  # 如果有视频文件路径
        playing = True
        extension = os.path.splitext(video_path)[1].lower()
        if extension in ['.avi', '.mp4']:
            # 在新的线程中启动视频处理，防止界面阻塞
            thread = threading.Thread(target=detect_video, args=(video_path,))
            thread.start()
    pass


button3 = tk.Button(root, text="开始检测！", command=start_detection, bg="red", fg="white", font=("Arial", 18, "bold"),
                    relief="raised", borderwidth=5)
button3.place(x=620, y=50)

root.mainloop()
