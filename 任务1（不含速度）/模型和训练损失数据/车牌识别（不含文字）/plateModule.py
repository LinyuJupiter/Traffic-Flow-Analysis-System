import torch
from torch import nn
import cv2
import numpy
from cnocr import CnOcr

chars = (
            u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣",
            u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
            u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5",
            u"6", u"7", u"8", u"9", u"A",
            u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"I", u"J", u"K", u"L", u"M", u"N", u"O", u"P", u"Q", u"R", u"S",
            u"T", u"U", u"V", u"W", u"X",
            u"Y", u"Z", u"港", u"学", u"使", u"警", u"澳", u"挂", u"军", u"北", u"南", u"广", u"沈", u"兰", u"成",
            u"济", u"海", u"民", u"航", u"临", u"领", u"null"
        )

char_dict = {c: i for i, c in enumerate(chars)}

char_len = len(chars)

class PlateOcr(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(16, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.gru = nn.GRU(1024, 256, 2, bidirectional=True,
                          batch_first=True)
        """
        -1: nn.GRU(1024, 256, 3
        -2: nn.GRU(1024, 256, 2
        -4: nn.GRU(1024, 256, 1
        """
        self.fc = nn.Linear(512, char_len)

    def forward(self, x) -> torch.tensor:
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, x.size(1), -1)
        x, _ = self.gru(x)
        x = self.fc(x)
        x = x.view(batch_size * x.size(1), -1)
        return x

    def predict(self, data) -> tuple[str, float]:
        if type(data) == numpy.ndarray:
            data = torch.tensor(data, dtype=torch.float)
            data = data.permute(2, 0, 1)
            data = data.view(-1, 3, 48, 128)
        output = self.forward(data)
        output = torch.softmax(output, dim=1)
        index = torch.argmax(output, dim=1)
        charList = [chars[i] for i in index]

        outString = ''
        confidence = 1.0
        for i, s in enumerate(charList):
            if s == 'null':
                continue
            else:
                outString += s
                confidence *= output[i][index[i]].item()
        return outString, confidence


if __name__ == '__main__':
    model = PlateOcr()
    model.load_state_dict(torch.load('ocrmodel\\ocr-4-2.pt'))
    img = cv2.imread('G:\\INeedChinese\\dataset\\CBLPRD-330k\\images\\000000060.jpg')
    out = model.predict(img)
    out2 = CnOcr().ocr(img)
    print(out)
    print(out2)
