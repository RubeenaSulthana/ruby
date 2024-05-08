## boundingbox
This Python script is designed to process a CSV file containing bounding box coordinates and associated image filenames. It then draws bounding boxes around objects in images and saves the annotated images with boxes overlaid, as well as cropped images of the individual objects.

## Libraries Used:
The scripts imports neccesary modules:

import os : Provides functions for interacting with the operating system.

import csv : Allows reading and writing CSV files.

from PIL import Image, ImageDraw : These are from the Python Imaging Library (PIL), which is used for opening, manipulating, and drawing on images.

## Defining path:

It defines paths for the CSV file (csv_file), the directory containing images (image_dir), and the directory where the output images will be saved (output_dir).

csv_file = "/home/rubeena-sulthana/Downloads/7622202030987_bounding_box.csv"

image_dir = "/home/rubeena-sulthana/Downloads/7622202030987/"

## Creating Output Directory:

It ensures that the output directory exists. If not, it creates it.

output_dir = "/home/rubeena-sulthana/Downloads/7622202030987_with_boxes"

os.makedirs(output_dir, exist_ok=True)

## Helper functions:

Two helper functions are defined:

## 1.draw boxes: 
This function takes an image and a list of dictionaries representing bounding boxes. 

It draws rectangles around the objects defined by the bounding boxes.

def draw_boxes(image, boxes):

    draw = ImageDraw.Draw(image)
    
    for box in boxes:
    
        left = int(box['left'])
        
        top = int(box['top'])
        
        right = int(box['right'])
        
        bottom = int(box['bottom'])
        
        draw.rectangle([left, top, right, bottom], outline="red")
        
    return image

## 2.crop image: 

This function crops the regions defined by the bounding boxes from the image.
    
def crop_image(image, boxes):

    cropped_images = []
    
    for box in boxes:
    
        left = int(box['left'])
        
        top = int(box['top'])
        
        right = int(box['right'])
        
        bottom = int(box['bottom'])
        
        cropped_img = image.crop((left, top, right, bottom))
        
        cropped_images.append(cropped_img)
        
    return cropped_images

## Note:

Make sure the CSV file follows the specified format with columns 'filename', 'xmin', 'ymin', 'xmax', 'ymax'.

Adjust the file paths (csv_file, image_dir, output_dir) according to your directory structure.

with open(csv_file, 'r') as file:

    csv_reader = csv.DictReader(file)
    
    for row in csv_reader:
    
        image_name = row['filename']
        
        image_path = os.path.join(image_dir, image_name)
        
        output_path = os.path.join(output_dir, image_name)
        
        image = Image.open(image_path)
        
        boxes = [{'left': row['xmin'], 'top': row['ymin'], 'right': row['xmax'], 'bottom': row['ymax']}]
        
        cropped_images = crop_image(image, boxes)
        
        for i, cropped_img in enumerate(cropped_images):
        
            cropped_img.save(os.path.join(output_dir, f"{i}_{image_name}"))  
            
        full_image_with_boxes = draw_boxes(image, boxes)
        
        full_image_with_boxes.save(os.path.join(output_dir, f"full_{image_name}"))
## here the input and output 





## histogram
This code is a Python script using OpenCV (cv2) and matplotlib libraries to read an image, compute its histogram, and plot it.

Let's break down the code step by step:

## Imports:

import numpy as np 

NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

import cv2 as cv

 OpenCV is a library of programming functions mainly aimed at real-time computer vision.

from matplotlib import pyplot as plt

 Matplotlib is a plotting library for Python.
 
 ## Reading and Writing the image:
 
img = cv.imread('/home/rubeena-sulthana/Desktop/experiment.s/ahad.jpeg')

This line reads an image file named "ahad.jpeg" from the specified path.

cv.imwrite("/home/rubeena-sulthana/Desktop/experiment.s/anki.png",img)

This line writes the read image (img) into another file named "anki.png" at the specified path.

assert img is not None, "file could not be read, check with os.path.exists()"
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
```

## task
```
num = list(range(10))
previousNum = 0
for i in num:
    sum = previousNum + i
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))
    previousNum=i
```

## videocp
```

# import the opencv library 
import cv2 
  
  
# define a video capture object 
vid = cv2.VideoCapture(0) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    cv2.imshow('frame', frame) 
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows()
```

