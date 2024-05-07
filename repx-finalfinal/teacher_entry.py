import requests
import PyPDF2
from PyPDF2 import PdfReader
import os
from PIL import Image

def download_pdf(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("PDF downloaded successfully.")
    else:
        print("Failed to download PDF.")

def download_image(url,code):
    img_data = requests.get(url).content
    with open(f"{code}/diagram_img.jpg", 'wb') as handler:
        handler.write(img_data)

    # sample_image = Image.open(f"{code}/diagram_img.jpg")
    # return sample_image

def create_assignment(code,model_ans,diagram_url):

    if os.path.exists(code):
        # print("Assignment already exists")
        return 0
    

    os.mkdir(code)

    url_model = model_ans
    save_path = f"{code}/downloaded_pdf.pdf"
    download_pdf(url_model, save_path)

    if(diagram_url!="none"):   
        download_image(diagram_url,code)

    return 1

# code=input("Enter assignment code:")

# create_assignment("003",
#                   "https://priyanshu-dutta-portfolio.vercel.app/assets/docs/model_ans.pdf",             #model ans
#                   "https://raw.githubusercontent.com/priyanshudutta04/priyanshudutta04/main/docs/Screenshot%2025.png")      #model diagram