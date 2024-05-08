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
##  input 
![7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/RubeenaSulthana/ruby/assets/169050067/459fc10a-b3eb-4a27-bb5e-77c8f4d7332d)

## Output 1

![full_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/RubeenaSulthana/ruby/assets/169050067/7863f459-faae-4ff6-b2d4-5efb3a214ed7)

## Output 2

![0_7622202030987_f306535d741c9148dc458acbbc887243_L_487](https://github.com/RubeenaSulthana/ruby/assets/169050067/bf819736-3124-4882-9ce8-b63ec2eb7adb)





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

## Input

![ahad](https://github.com/RubeenaSulthana/ruby/assets/169050067/4e2f8328-a1d4-478a-a2e0-fc59fca20894)


This line reads an image file named "ahad.jpeg" from the specified path.

cv.imwrite("/home/rubeena-sulthana/Desktop/experiment.s/anki.png",img)

This line writes the read image (img) into another file named "anki.png" at the specified path.

## Assertion:

assert img is not None, "file could not be read, check with os.path.exists()"

This line checks if the image has been read successfully. If the image reading fails, it raises an AssertionError with the given message.

## Histogram Calculation and Plotting:

color = ('b','g','r')

This line defines a tuple of colors to be used for plotting histograms for each channel: blue, green, and red.

for i,col in enumerate(color):

 histr = cv.calcHist([img],[i],None,[256],[0,256])

 This calculates the histogram of the image img for the current color channel (i), with 256 bins in the range 0 to 255.
 
 plt.plot(histr,color = col)
 
 This plots the histogram using pyplot.plot(), where histr is the histogram data and col is the color corresponding to the channel.
 
 plt.xlim([0,256])

 This sets the x-axis limit of the plot to be between 0 and 256, which represents the intensity values.
 
plt.show()
This displays the histogram plot.

## Output:

![anki2](https://github.com/RubeenaSulthana/ruby/assets/169050067/222a95eb-4374-4958-99a6-53e553d32297)



## Iterating the first 10 numbers
This Python code defines a list of numbers from 0 to 9 (num). It then iterates over each number in this list.

num = list(range(10))

This line creates a list called num containing numbers from 0 to 9.

previousNum = 0

Initializes a variable called previousNum with the value 0.

for i in num:

This is a loop that iterates over each element in the list num.
Inside the loop:

    sum = previousNum + i
    Adds the current value of i to the previousNum and stores it in a variable called sum.
    
    print('Current Number '+ str(i) + 'Previous Number ' + str(previousNum) + 'is ' + str(sum))

    Prints the current number 'i', the previous number 'previousNum', and their sum 'sum'.
    previousNum=i
    Prints the current number i, the previous number previousNum, and their sum sum.
So, the code calculates the sum of each number in the list num with its previous number and prints them.
## Output:
Current Number 0Previous Number 0is 0

Current Number 1Previous Number 0is 1

Current Number 2Previous Number 1is 3

Current Number 3Previous Number 2is 5

Current Number 4Previous Number 3is 7

Current Number 5Previous Number 4is 9

Current Number 6Previous Number 5is 11

Current Number 7Previous Number 6is 13

Current Number 8Previous Number 7is 15

Current Number 9Previous Number 8is 17

## web cam
This Python code snippet utilizes the OpenCV library to capture video from a webcam (or other video input device) and display it in a window. 
Let's break down the code step by step:

1.import the opencv library 

import cv2 

2.define a video capture object (vid) by calling cv2.videocapture(0)

vid = cv2.VideoCapture(0) 

3.Start an infinite loop to continuously capture and display frames:

while(True): 

4. the function vid.read() captures a frame from the video and returns two values:ret,a boolean indicating whether a frame was successfully captured, and frame, the captured frame.
   
    ret, frame = vid.read() 
  
5.Display the resulting frame 

    cv2.imshow('frame', frame) 
      
6.The loop waits for a key press with cv2.waitKey(1).If the pressed key is 'q'.the loop breaks and the program ends.

    if cv2.waitKey(1) & 0xFF == ord('q'): 
    
        break
  
7.vid.release() releases the video captured object, freeing the camera resources.

vid.release() 

8.cv2.DestroyAlltheWindows() closes all OpenCV windows.

cv2.destroyAllWindows()

## Output



