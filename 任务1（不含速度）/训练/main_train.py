from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(torch.cuda.device_count())
    model = YOLO('G:\\INeedChinese\\runs\\detect\\train11\\weights\\car_clp_1.pt')
    model.train(data='BITVehicle.yaml',
                workers=1,
                device='cuda:0',
                epochs=8,
                batch=8,
                lr0=0.01,
                lrf=0.1,
                val=True,
                pretrained=True,
                )
