from PIL import Image, ImageDraw, ImageFont
import pandas as pd

# data used
data = pd.read_csv("data.csv")

# size and color of text
color = (0, 0, 0)
font_size_name = 100
font_size_other = 38

# (width, height)
name_location = (700, 770)
year_location = (250, 885)
class_location = (1150, 885)
position_location = (800, 955)

# font used
font_used_name = "./font1.ttf"
font_used_other = "./font2.ttf"

# pic to be edited
pic = "cert.png"

# output folder
save_folder = "certs"

# program
class_list = data["class"].tolist()
year_list = data["year"].tolist()
pos_list = data["position"].tolist()
for i, name in enumerate(data["name"]):
    img = Image.open(pic)
    d = ImageDraw.Draw(img)
    font_name = ImageFont.truetype(font_used_name, font_size_name, encoding="unic")
    font_other = ImageFont.truetype(font_used_other, font_size_other, encoding="unic")
    d.text(name_location, name, fill=color, font=font_name)
    d.text(class_location, class_list[i], fill=color, font=font_other)
    d.text(year_location, year_list[i], fill=color, font=font_other)
    d.text(position_location, pos_list[i], fill=color, font=font_other)
    img.convert("RGB").save(f"./{save_folder}/{name}.pdf")
