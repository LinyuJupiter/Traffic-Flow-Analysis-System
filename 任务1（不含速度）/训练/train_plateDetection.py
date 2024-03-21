from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.device_count())
    model = YOLO('G:\\INeedChinese\\runs\\detect\\train28\\weights\\best.pt')
    model.train(data='plate.yaml',
                device='cuda:0',
                workers=2,
                epochs=1,
                batch=2,
                imgsz=258,
                val=False,
                )
