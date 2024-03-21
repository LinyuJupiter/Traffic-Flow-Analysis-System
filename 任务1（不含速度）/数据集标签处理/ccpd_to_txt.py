import os

image_folder = "./images"

label_folder = "./labels/"

width, height = 720, 1160

if not os.path.exists(label_folder):
    os.makedirs(label_folder)

for image_file in os.listdir(image_folder):
    if image_file.endswith(".jpg"):
        file_name = os.path.splitext(image_file)[0]
        _, _, box, _, _, _, _ = file_name.split("-")

        lu, rb = box.split('_')
        lu_x, lu_y = lu.split('&')
        rb_x, rb_y = rb.split("&")

        lu_x = float(lu_x)
        lu_y = float(lu_y)
        rb_x = float(rb_x)
        rb_y = float(rb_y)

        # Calculate the center, width and height of the LP bounding box
        center_x = (lu_x + rb_x) / 2 / width
        center_y = (lu_y + rb_y) / 2 / height
        bbox_w = (rb_x - lu_x) / width
        bbox_h = (rb_y - lu_y) / height


        label_file = open(label_folder + file_name + ".txt", "w")

        label_file.write("0 {} {} {} {}\n".format(center_x, center_y, bbox_w, bbox_h))

        label_file.close()
