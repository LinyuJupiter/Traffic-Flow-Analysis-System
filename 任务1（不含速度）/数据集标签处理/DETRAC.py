from xml.etree import ElementTree as et
import os

vehicleDict = {'car': '0', 'SUV': '1', 'bus': '2', 'taxi': '3', 'truck': '4', 'van': '5', 'others': '6'}
root_path = 'G:\\INeedChinese\\source\\train_video\\DETRAC-Train-Annotations-XML'
for xml_file in os.listdir(root_path):
    root = et.parse(os.path.join(root_path, xml_file))
    for frame in root.iter(tag='frame'):
        txtFile = open(f'G:\\INeedChinese\\dataset\\DETRAC-train-data\\labels\\{xml_file[:-4]}_{frame.attrib["num"]}.txt', 'w')
        for target in frame.iter(tag='target'):
            x, y, w, h = target[0].attrib.values()
            x = (float(x) + float(w) / 2) / 960
            y = (float(y) + float(h) / 2) / 540
            outStr = f'{vehicleDict[target[1].attrib["vehicle_type"]]} {x} {y} {float(w)/960} {float(h)/540}'
            txtFile.write(outStr + '\n')
        txtFile.close()
