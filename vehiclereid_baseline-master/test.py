import torch.utils.data.distributed
import torchvision.transforms as transforms

from PIL import Image
from models import *


def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 打开图像并转换为RGB格式
    preprocess = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图片归一化
                             std=[0.229, 0.224, 0.225])
    ])
    preprocessed_image = preprocess(image)
    preprocessed_image = preprocessed_image.unsqueeze(0)  # 添加额外的批次维度
    return preprocessed_image


# gpu or not
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cnn.CNN(num_classes=57).to(device)
state_dict = torch.load(r"./checkpoints/att/Car_epoch_15.pth")
model.load_state_dict(state_dict['state_dict'])
model.eval()
img = load_and_preprocess_image(r"./000.jpg")
with torch.no_grad():
    outputs = model(img)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # 应用 softmax 函数获得概率
index = torch.argmax(probabilities[0]).tolist()
print(index)
print(probabilities[0][index])

# 将probabilities[0][index]从tensor转为list
probabilities = probabilities[0][index].tolist()
# 打印刚刚的结果
print(probabilities)