import streamlit as st
from PIL import Image
import numpy as np
import os
import torch
from ultralytics import YOLO


model_path = r'C:\Users\Zeus\PycharmProjects\datascienceapp\5_conditions\model\5_conditions_best.pt'
model = YOLO(model_path)

def object_detection_image():
    st.title('Screening of Chest X-ray Images')

    input_method = st.selectbox('Select Image Input Method', ('Upload from Gallery', 'Take a Picture'))

    if input_method == 'Upload from Gallery':
        file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    else:
       file = st.camera_input('Take a picture of the image')

    if file is not None:
        img1 = Image.open(file)
        img2 = np.array(img1)
        # img1_1 = cv2.imread(file)

        st.image(img1, caption="Uploaded Image")
        my_bar = st.progress(0)


        # Perform object detection using YOLOv5
        results = model.predict(img2, save= True, conf = 0.2, batch =2)

        ## path to results (this will alwys get the latest detection

        path = "C:/Users/Zeus/PycharmProjects/datascienceapp/5_conditions/runs/detect"
        detection = (os.listdir("C:/Users/Zeus/PycharmProjects/datascienceapp/5_conditions/runs/detect"))



        def extract_digits(x):
            digits = ''.join(filter(str.isdigit, x))
            return int(digits) if digits else float('inf')  # Use float('inf') as a default value

        sorted_ = sorted(detection, key=extract_digits)

        if len(sorted_)>1:
            latest_detection = sorted_[-2]
        else:
            latest_detection = sorted_[0]

        detection_path = os.path.join(path, latest_detection)


        st.image(Image.open(os.path.join(detection_path, 'image0.jpg')), caption= "Predictions")


        ## To get the classes of the predictions

        all_predictions = []
        for result in results:
            boxes = result.boxes
            boxes = boxes.cls.cpu().numpy().tolist()
            all_predictions.append(boxes)

        conditions = ['Aortic enlargement', 'Atelectasis', 'Calcification', 'Cardiomegaly', 'Consolidation', 'ILD',
                      'Infiltration', 'Lung Opacity', 'Nodule', 'Other lesions', 'Pleural effusion',
                      'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']


        ###Printing out the conditions present
        # #
        positive = []
        for pred in all_predictions:
           for sub_pred in pred:
                pres =(conditions[int(sub_pred)])  ## converted to a set to remove duplicates and converted back to a list
                positive.append(pres)

        ## Display the results
        st.header('Screening results')

        # st.write(boxes)

        if len(positive)>0:
            for pos in list(set(positive)):
                st.write(pos)
        else:
            st.write('No conditions detected')

#
def main():
    new_title = '<p style="font-size: 42px;">Your Radiologist Assistant 🧑🏾‍⚕️🩻️</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    At DelΦ, we are on a mission to make healthcare more efficient, cost-effective, and inclusive. 
    With a dedicated team of experts in both medical and AI fields, we have developed a state-of-the-art web platform 
    that leverages the capabilities of Artificial Intelligence to analyze chest X-ray images for 5 common chest conditions:
    Cardiomegaly, Pleural effusion, Lung opacity, Nodule and Pneumothorax.
    
    The Future of Healthcare:
    Our AI-driven chest X-ray screening web application is just the beginning. 
    We envision a future where AI is a valuable tool in the hands of healthcare providers, 
    improving diagnostic accuracy, reducing costs, and ultimately saving lives.
    """
                          )
    ## front page
    desk = Image.open(r"C:\Users\Zeus\PycharmProjects\datascienceapp\5_conditions\media\frontpage.jpeg")
    front_page = st.image(desk, width= 700)
    st.sidebar.title("DelΦ")
    choice = st.sidebar.selectbox("MODE", ("About", "Screen CXR"))

    if choice == "Screen CXR":
        read_me_0.empty()
        read_me.empty()
        front_page.empty()
        object_detection_image()

    elif choice == "About":
        print()

if __name__ == '__main__':
    main()
