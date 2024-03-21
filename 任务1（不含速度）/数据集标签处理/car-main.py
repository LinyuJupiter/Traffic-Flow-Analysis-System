from xml.etree import ElementTree as et
import os

vehicleDict = {'car': '0', 'SUV': '1', 'bus': '2', 'taxi': '3', 'truck': '4', 'van': '5', 'others': '6'}
root_path = 'G:\\INeedChinese\\dataset\\car-main\\annotation'
labels_path = 'G:\\INeedChinese\\dataset\\car-main\\labels'

for xml_file in os.listdir(root_path):
    root = et.parse(os.path.join(root_path, xml_file))
    txt_name = root.find('filename').text
    txt_name = txt_name[:-3] + 'txt'

    vehicle = root.find('object').find('name').text

    box = root.find('object').find('bndbox')
    x, y, w, h = box.find('xmin').text, box.find('ymin').text, box.find('xmax').text, box.find('ymax').text
    x, y, w, h = float(x), float(y), float(w), float(h)
    w, h = w-x, h-y
    x, y = x + 0.5*w, y + 0.5*h
    size = root.find('size')
    xlim, ylim = float(size.find('width').text), float(size.find('height').text)
    x, y, w, h = x/xlim, y/ylim, w/xlim, h/ylim
    with open(os.path.join(labels_path, txt_name), 'w') as txt_file:
        txt_file.write(f"{vehicleDict[vehicle]} {x} {y} {w} {h}")
