import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

import PyPDF2
from PyPDF2 import PdfReader
from io import BytesIO
import os
import fitz
import requests

from PIL import Image
import math
from math import sqrt
import cv2

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True,lang='en')

from textblob import TextBlob
from spellchecker import SpellChecker
import re

import spacy
import nltk
nltk.download('punkt')
import nltk.corpus

def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download PDF.")

def download_image(url):
    img_data = requests.get(url).content
    with open('diagram_img.jpg', 'wb') as handler:
        handler.write(img_data)

    sample_image = Image.open('diagram_img.jpg')
    return sample_image


def extract_text_from_pdf(save_path):
    pdf_document = fitz.open(save_path)
    text = ''
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text += page.get_text() + "\n"
    return text
    

def pdf_to_images(pdf_path, output_folder, resolution=300):

    images=[]

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)


    for page_number in range(len(pdf_document)):
        # Get the page
        page = pdf_document[page_number]

        matrix = fitz.Matrix(resolution/72, resolution/72)
        image = page.get_pixmap(matrix=matrix)

        output_image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")

        image.save(output_image_path)
        images.append(output_image_path)

    pdf_document.close()
    return images

def separate_diagram(image_path,threshold):

    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # contour with the largest area (likely the diagram)
    max_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(max_contour)

    if max_area<threshold:
      # print("No Diagram")
      return None
    # print(max_area)

    x, y, w, h = cv2.boundingRect(max_contour)
    side_length = max(w, h)

    # Ensure the square fits within the image dimensions
    side_length = min(side_length, min(image.shape[:2]))

    # Crop
    cropped_diagram = image[y:y + side_length, x:x + side_length]

    return cropped_diagram


def cosineSim(a1,a2):
    sum = 0
    suma1 = 0
    sumb1 = 0
    for i,j in zip(a1, a2):
        suma1 += i * i
        sumb1 += j*j
        sum += i*j
    cosine_sim = sum / ((sqrt(suma1))*(sqrt(sumb1)))
    return cosine_sim

def check(res):
    str=re.findall("[a-zA-Z]+",res)
    updated_res=(" ".join(str))
    spell = SpellChecker()
    misspelled = spell.unknown(str)

    if len(updated_res)==0:
        return 100000                           # if len is 0, avoid this case
    return(len(misspelled)/len(updated_res))      # ratio of errors to word


def correct(text):
  # txt = TextBlob(text)
  # return txt.correct()
  spell = SpellChecker()
  words = text.split()
  corrected_text = []

  for word in words:
      corrected_word = spell.correction(word)

      if corrected_word is None:
          corrected_word = word

      corrected_text.append(corrected_word)

  corrected_text = " ".join(corrected_text)
  return corrected_text


def text_to_img(file_name):
  image = cv2.imread(file_name)
  h, w = image.shape[:2]
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Sum white pixels in each row
  # Create blank space array and and final image
  pixels = np.sum(thresh, axis=1).tolist()
  space = np.ones((4, w), dtype=np.uint8) * 255
  result = np.zeros((0, w), dtype=np.uint8)

  # Iterate through each row and add space if entire row is empty
  # otherwise add original section of image to final image
  for index, value in enumerate(pixels):
      if value == 0:
          result = np.concatenate((result, space), axis=0)
      row = gray[index:index+1, 0:w]
      result = np.concatenate((result, row), axis=0)


  img_read=result

  # global thresholding
  retval, img_thresh1=cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
  retval, img_thresh2=cv2.threshold(img_read, 120, 255, cv2.THRESH_BINARY)

  # adaptive thresholding
  img_thresh_adap=cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,7)

  result1 = ocr.ocr(img_read)
  result2 = ocr.ocr(img_thresh1)
  result3 = ocr.ocr(img_thresh2)
  result4 = ocr.ocr(img_thresh_adap)

  str1=""
  if result1[0] is not None:
    for i in range(len(result1[0])):
      str1=str1+result1[0][i][1][0]+" "


  print()
  str2=""
  if result2[0] is not None:
    for i in range(len(result2[0])):
      str2=str2+result2[0][i][1][0]+" "


  print()
  str3=""
  if result3[0] is not None:
    for i in range(len(result3[0])):
      str3=str3+result3[0][i][1][0]+" "


  print()
  str4=""
  if result4[0] is not None:
    for i in range(len(result4[0])):
      str4=str4+result4[0][i][1][0]+" "


  best=""
  minm = min(check(str1), check(str2), check(str3), check(str4))

  if minm==check(str1):
    best=str1
  elif minm==check(str2):
    best=str2
  elif minm==check(str3):
    best=str3
  else:
    best=str4

  return correct(best)


#program run
def get_answer( scan_ans,code):

    # url_model = model_ans
    # save_path = "downloaded_pdf.pdf"
    # download_pdf(url_model, save_path)
    if not os.path.exists(code):
        print("Error: No such assignment exists")
        return

    model_ans = extract_text_from_pdf(f"{code}/downloaded_pdf.pdf")

    url_scan = scan_ans
    save_path_scan = f"{code}/scan.pdf"
    download_pdf(url_scan, save_path_scan)

    output_folder =  f"{code}/output_images"
    images=pdf_to_images(save_path_scan, output_folder, 300)


    sample_image = Image.open(f"{code}/diagram_img.jpg")
    img1 = sample_image.convert('L')

    isdiagram=False

    for i in range(len(images)):
        diagram = separate_diagram(images[i],5000)
        if diagram is not None:
            isdiagram=True
            break

    diagram_marks=0

    if isdiagram:
        img_arr1 = np.array(diagram)
        img_arr2 = np.array(img1)

        flat_img1=img_arr1.flatten()
        flat_img2=img_arr2.flatten()

        # normalizing
        norm_img1 = flat_img1/255.0
        norm_img2 = flat_img2/255.0

        similarity=cosineSim(norm_img1,norm_img2)*100
        # print("Diagram Similarity:",end=" ")
        # print(round(similarity, 2))

        if round(similarity, 2)>0.95:
            diagram_marks=5
        elif round(similarity, 2)>0.9:
            diagram_marks=4
        elif round(similarity, 2)>0.85:
            diagram_marks=3
        elif round(similarity, 2)>0.8:
            diagram_marks=2
        elif round(similarity, 2)>0.75:
            diagram_marks=1
        else:
            diagram_marks=0

    
    eval_text=""

    for i in range(len(images)):
        txt=text_to_img(images[i])
        eval_text=eval_text+txt+" "

    # print(eval_text)

    #Deleting student data!
    shutil.rmtree(f"{code}/output_images")
    os.remove(f"{code}/scan.pdf")
        
    str=re.findall("[a-zA-Z]+",eval_text)
    updated_ans=(" ".join(str))
    print(updated_ans)

    nlp=spacy.load("en_core_web_lg")
    v1=nlp(model_ans)
    v2=nlp(updated_ans)

    sim=v1.similarity(v2)
    sim

    marks=0
    if sim>0.99:
        marks=10
    elif sim>0.98:
        marks=9
    elif sim>0.97:
        marks=8
    elif sim>0.95:
        marks=7
    elif sim>0.92:
        marks=6
    elif sim>0.90:
        marks=4
    elif sim>0.87:
        marks=3
    elif sim>0.82:
        marks=2
    elif sim>0.78:
        marks=1
    else:
        marks=0

    total_marks=diagram_marks+marks

    if isdiagram:
        print("Total Marks out of 15: ",total_marks)
    else:
        print("Total Marks out of 10: ",total_marks)

    return total_marks

code=input("Enter assignment code:")

get_answer(
            "https://priyanshu-dutta-portfolio.vercel.app/assets/docs/Scan_2024.pdf",                               #student answer
            code       #code
            )

