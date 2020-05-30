#!/usr/bin/env python
# coding: utf-8

# # Problem 2 Submission

# ### Note: I have written this algothm with the constraints like the number plate can have only either one or two lines data.

import xml.etree.ElementTree as ET 
import numpy as np


image_list = ["./char_detection/AP03TC9939.xml", 
              "./char_detection/AP03TE2219.xml",
              "./char_detection/AP04V5222-6.xml",
              "./char_detection/AP35X9339-2.xml",
              "./char_detection/JH12G9561-3.xml"
             ]


def form_image_tree(img_path):
    tree = ET.parse(img_path)
    root = tree.getroot()
    return root


def form_tree_lists(root):
    x_min=[]
    x_max=[]
    y_min=[]
    y_max=[]

    for xmin in root.iter('xmin'):
        x_min.append(int(xmin.text))

    for xmax in root.iter('xmax'):
        x_max.append(int(xmax.text))

    for ymin in root.iter('ymin'):
        y_min.append(int(ymin.text))

    for ymax in root.iter('ymax'):
        y_max.append(int(ymax.text)) 
        
    print('X-min: ', x_min)
    print('X-max: ', x_max)
    print('Y-min: ', y_min)
    print('Y-max: ', y_max)

    return [x_min, y_min, x_max, y_max]


def calculate_main_arr(root):
    main_arr = []

    for i in range(len(list(root.iter('name')))):
        name = list(root.iter('name'))[i].text
        xmin = int(list(root.iter('xmin'))[i].text)
        xmax = int(list(root.iter('xmax'))[i].text)
        ymin = int(list(root.iter('ymin'))[i].text)
        ymax = int(list(root.iter('ymax'))[i].text)
        main_arr.append([name, xmin, ymin, xmax, ymax])
    return main_arr


def sort_nd_list(sub_li, element_pos): 
    l = len(sub_li) 
    for i in range(0, l): 
        for j in range(0, l-i-1): 
            if (sub_li[j][element_pos] > sub_li[j + 1][element_pos]): 
                tempo = sub_li[j] 
                sub_li[j]= sub_li[j + 1] 
                sub_li[j + 1]= tempo 
    return sub_li 


def is_double_line(arr, y_min_mean):
    for i in arr:
        if abs(i[2] - y_min_mean) < 10:
            return False
    return True


def find_sorted_sequence(arr, coords):
    
    upper_line = []
    lower_line = []
    
    y_min = coords[1]
    y_min_mean = np.mean(y_min)
    print("Mean Y: ", y_min_mean)
    
    is_double = is_double_line(arr, y_min_mean)
    print("Double Line: ", is_double)
        
    for i in range(len(arr)):
        if is_double:
            if arr[i][2] > y_min_mean:
                lower_line.append(arr[i])
            else:
                upper_line.append(arr[i])
                
            sorted_upper = sort_nd_list(upper_line, 1)                
            sorted_lower = sort_nd_list(lower_line, 1)
            
        else:
            upper_line.append(arr[i])
            lower_line = []

            sorted_upper = sort_nd_list(upper_line, 1)                
            sorted_lower = sort_nd_list(lower_line, 1)
            

    print('='*20)
    print("Sorted upper list: ")
    print(sorted_upper, end="\n\n")
    print("Sorted lower list: ")
    print(sorted_lower)
    return [sorted_upper, sorted_lower]


def print_sorted_numer_plate(upper, lower):
    print("\nFINAL SORTED CHARACTERS:")
    for i in upper:
        print(i[0], end="")
    for i in lower:
        print(i[0], end="")
    print()
    print('='*10, end='\n\n\n\n')


for i in image_list:
    print('='*80)
    root = form_image_tree(i)
    coord_list = form_tree_lists(root)
    main_list = calculate_main_arr(root)
    upper, lower = find_sorted_sequence(main_list, coord_list)
    print_sorted_numer_plate(upper, lower)
    print('='*80)


# ### end

# <hr><hr><hr>

# ### Image Detection (not used here)

# In[14]:


# !pip install --upgrade google-cloud-vision


# In[15]:


# import os
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="D:/main_jupyter_workspace/tarsyer/gcp_key.json"


# In[16]:


# from google.cloud import vision
# from google.cloud.vision import types
# import re


# In[17]:


# pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# In[18]:


# im = Image.open("./char_detection/AP03TC9939.jpg")
# im2 = im.crop(main_arr[1][1:])
# pppp = np.array(im2)

# img = Image.fromarray(pppp)
# img.show()

# print(np.array(im2))
# im2.save('c1.jpg', 'JPEG')


# In[19]:


# client = vision.ImageAnnotatorClient()
# image = vision.types.Image()


# def extract_text(file_name):
#     folder_path = 'D:/main_jupyter_workspace/tarsyer'
#     with io.open(os.path.join(folder_path, file_name), 'rb') as img_file:
#         content = img_file.read()
        
#     img = vision.types.Image(content=content)
#     response = client.text_detection(image=img)
    
#     label = response.text_annotations[0].description
#     return str(label)


# In[20]:


# images_names = ['c1.jpg']
# response_arr = []
# for img in images_names:
#     response_arr.append(extract_text(img))
    
# num_plate_str = response_arr[0]
# print(re.sub(r"\W", "", num_plate_str))


# In[21]:


# def find_text(img): 
#     text=pytesseract.image_to_string(img, config="--psm 6")
#     print(text)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     gray = cv2.bitwise_not(gray)

#     kernel = np.ones((2, 1), np.uint8)
#     img = cv2.erode(img, kernel, iterations=1)
#     img = cv2.dilate(img, kernel, iterations=1)
#     out_below = pytesseract.image_to_string(img, config="--psm 6")
#     print("-"*20)
#     print("OUTPUT:", out_below)
#     img = Image.fromarray(img)
#     img.show()


# In[22]:


# img=cv2.imread('c1.jpg')
# find_text(pppp)

