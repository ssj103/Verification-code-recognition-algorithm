# -*- coding: utf-8 -*-

import sys
import os
import shutil
import random
import time

## Captcha is a library for generating captcha images, which can be installed by pip install captcha
from captcha.image import ImageCaptcha

import matplotlib.pyplot as plt
from PIL import Image
 
## char
CHAR_SET = ['0','1','2','3','4','5','6','7','8','9']
CHAR_SET_LEN = 10
CAPTCHA_LEN = 4
 
## train images save path
CAPTCHA_IMAGE_PATH = 'captcha/images/'
## test images save path
TEST_IMAGE_PATH = 'captcha/test/'
## test images number
TEST_IMAGE_NUMBER = 2000
 
## Generate a captcha image, 4 digits of decimal digits can have 10,000 captchas
def generate_captcha_image(charSet = CHAR_SET, charSetLen=CHAR_SET_LEN, captchaImgPath=CAPTCHA_IMAGE_PATH):   
    k  = 0
    total = 1
    for i in range(CAPTCHA_LEN):
        total *= charSetLen
        
    for i in range(charSetLen):
        for j in range(charSetLen):
            for m in range(charSetLen):
                for n in range(charSetLen):
                    captcha_text = charSet[i] + charSet[j] + charSet[m] + charSet[n]
                    image = ImageCaptcha()
                
                    
                    image.write(captcha_text, captchaImgPath + captcha_text + '.jpg')
                    
                    img = Image.open(captchaImgPath)
                    
                    print("image = ",image)
                    #转为灰度图
                    img = image.convert("L")     
                    print("img")
                    plt.figure("img")
                    plt.imshow(img)
                    plt.show()
                    
                    k += 1
                    sys.stdout.write("\rCreating %d/%d" % (k, total))
                    sys.stdout.flush()
                    
## Take a part of the image set from the verification code as a test set.
## These pictures are not used for training and are only used for model testing.                   
def prepare_test_set():
    fileNameList = []    
    for filePath in os.listdir(CAPTCHA_IMAGE_PATH):
        captcha_name = filePath.split('/')[-1]
        fileNameList.append(captcha_name)
    random.seed(time.time())
    random.shuffle(fileNameList) 
    for i in range(TEST_IMAGE_NUMBER):
        name = fileNameList[i]
        shutil.move(CAPTCHA_IMAGE_PATH + name, TEST_IMAGE_PATH + name)
                        
if __name__ == '__main__':
    generate_captcha_image(CHAR_SET, CHAR_SET_LEN, CAPTCHA_IMAGE_PATH)
    prepare_test_set()
    sys.stdout.write("\nFinished")
    sys.stdout.flush()  