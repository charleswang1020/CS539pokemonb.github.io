import os
import cv2
from PIL import Image


src = "./Charmander/Charmander" 
dst = "./Charmander/Charmander1"

os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)

src = "./Pikachu/Pikachu" 
dst = "./Pikachu/Pikachu1"

os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)

src = "./Squirtle/Squirtle" 
dst = "./Squirtle/Squirtle1"

os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)

src = "./Pikachu/Pikachu1/"
dst = "./Pikachu/Pikachu2/"

os.mkdir(dst)

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    if png.mode == 'RGBA':
        png.load()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) 
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')

src = "./Charmander/Charmander1/"
dst = "./Charmander/Charmander2/"

os.mkdir(dst)

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    if png.mode == 'RGBA':
        png.load()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) 
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')

src = "./Squirtle/Squirtle1/"
dst = "./Squirtle/Squirtle2/"

os.mkdir(dst)

for each in os.listdir(src):
    png = Image.open(os.path.join(src,each))
    if png.mode == 'RGBA':
        png.load()
        background = Image.new("RGB", png.size, (0,0,0))
        background.paste(png, mask=png.split()[3]) 
        background.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')
    else:
        png.convert('RGB')
        png.save(os.path.join(dst,each.split('.')[0] + '.jpg'), 'JPEG')