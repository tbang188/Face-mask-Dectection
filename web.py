import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import av
import cv2
import os
from datetime import datetime

import pandas as pd
import pydeck as pdk
import requests
import streamlit as st
from PIL import Image
from requests.exceptions import ConnectionError
import plotly.graph_objects as go
from datetime import date,datetime

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import detect_mask_image


# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(page_title='Face Mask Detector', page_icon='😷', layout='centered', initial_sidebar_state='expanded')

def config():

    # code to check turn of setting and footer
    st.markdown(""" <style>
    MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style> """, unsafe_allow_html=True)

    # encoding format
    encoding = "utf-8"

    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-color: #1c4b27;
            }
        </style>""",
        unsafe_allow_html=True,
    )

    st.balloons()
    # I want it to show balloon when it finished loading all the configs


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def list_of_countries():
    df = pd.read_csv("./csv/countries.csv")
    return df["Name"].tolist()


def covid_data_menu():
    st.subheader('Covid Data Menu')
    col1, col2, col3 = st.columns([4, 4, 4])
    with col1:
        st.text_input(label="Last Updated", value=str(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")), disabled=True)
    with col2:
        pass
    with col3:
        try:
            url = "https://disease.sh/v3/covid-19/countries"
            response = requests.get(url)
            countries = [i.get("country") for i in response.json()]
            option = st.selectbox('please select country?', (countries), help="Please select country")


        except ConnectionError:
            st.error("There is a connection error we failed to fetch all the countries 😥")
    try:
        response = requests.get("https://disease.sh/v3/covid-19/countries/" + option)
        data = response.json()

        col1, col2 = st.columns([6, 6])
        with col1:
            st.write("Country Info")
            country_data = data.pop("countryInfo")
            longitude, latitude = country_data["long"], country_data["lat"]
            country_data.update({"country": data["country"]})
            country_data.pop("lat")
            country_data.pop("long")
            # df = pd.DataFrame.from_dict(country_data, orient="index", dtype=str, columns=['Value'])
            # st.dataframe(df)
            remote_css("")
            st.markdown(f"""
               <table class="table table-borderless">
                    <tr>
                      <td>country</td>
                      <td>{country_data["country"]}</td>
                    </tr>
                     <tr>
                      <td>flag</td>
                      <td><img src="{country_data["flag"]}" style="width:20%;height:40%"></td>
                    </tr>
                    <tr>
                      <td>iso2</td>
                      <td>{country_data["iso2"]}</td>
                    </tr>
                    <tr>
                      <td>iso3</td>
                      <td>{country_data["iso3"]}</td>
                    </tr>
               </table></br>
            """, unsafe_allow_html=True)

            st.write("Covid Statistics")
            data.pop("country")
            data['updated'] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            df = pd.DataFrame.from_dict(data, orient="index", dtype=str, columns=['Value'])
            st.write(df)

        with col2:
            st.write("Map")
            st.pydeck_chart(pdk.Deck(
                map_style='mapbox://styles/mapbox/light-v9',
                initial_view_state=pdk.ViewState(
                    latitude=latitude,
                    longitude=longitude,
                    zoom=4.7,
                    pitch=50,
                )
            ))

        st.subheader("Vaccination Data")
        current_date = datetime.today().date()
        first_day_of_month = current_date.replace(day=1)
        number_of_days = (date.today() - first_day_of_month).days

        url = "https://disease.sh/v3/covid-19/vaccine/coverage/countries?lastdays=" + str(number_of_days)
        response = requests.get(url)
        vaccination_data = {}
        for i in response.json():
            if i.get("country") == option:
                vaccination_data = i.get("timeline")

        if len(vaccination_data) != 0:
            vaccination_data = {str(key): str(value) for key, value in vaccination_data.items()}
            st.write(vaccination_data)
            df = pd.DataFrame({'date': vaccination_data.keys(), 'vaccination_value': vaccination_data.values()})
            trace = go.Bar(x=df['date'], y=df['vaccination_value'], showlegend=True)
            layout = go.Layout(title=option)
            data = [trace]
            fig = go.Figure(data=data, layout=layout)
            st.plotly_chart(fig)
        else:
            st.write("Vaccination data for %s no available" % option)

        with st.expander('Covid 19 Prevention Tips'):
            st.subheader("Here’s what you can do to protect yourself:")
            st.markdown(f"""<p>At International Medical Corps, we’re always preparing for the unexpected—whether it’s
            an earthquake, a hurricane or an outbreak of infectious disease. As the COVID-19 outbreak grows,
            it’s important to know that there are many actions we can take to protect ourselves, our loved ones and
            our communities.</p>""", unsafe_allow_html=True)

            st.subheader("Here’s what you can do to protect yourself:")
            st.markdown(f""" <ul> <li>Wash your hands frequently with soap and water for at least 20 seconds.</li>
            <li>If soap and water are not available, use an alcohol-based hand sanitizer with at least 60%
            alcohol.</li> <li>Avoid close contact with people who are sick.</li> <li>Especially if you’re in a
            high-risk group, consider limiting your exposure to others, using social distancing—for example,
            avoid large gatherings, crowds of people and frequent trips to the store.</li>
            </li>Visit your state and local public-health websites for additional guidance specific to your area.</li>
             <li>Those at higher risk for serious illness should take additional precautions.</li>
              </ul> """, unsafe_allow_html=True)

            st.markdown(
                f"""</br> Reference for Tips : <a href="https://internationalmedicalcorps.org/emergency-response/covid-19/coronavirus-prevention-tips/">IMC</a>""",
                unsafe_allow_html=True)
    except ConnectionError as e:
        st.error("There is a connection error please retry later 😥")



def local_css(file_name):
    """ Method for reading styles.css and applying necessary changes to HTML"""
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

def mask_image():
    global RGB_img
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    model = load_model("mask_detector.model")

    # load the input image from disk and grab the image spatial
    # dimensions
    image = cv2.imread("./images/out.jpg")
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # pass the face through the model to determine if the face
            # has a mask or not
            (mask, withoutMask) = model.predict(face)[0]

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


mask_image()



def mask_detection():
    # class VideoTransformer(VideoTransformerBase):
    #     def __init__(self) -> None:
    #         (self.net, self.model) = mask_image_init()

    #     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
    #         image = frame.to_ndarray(format="bgr24")
    #         detections = get_detections_from_image(self.net, image)
    #         RGB_img = mask_image(self.model, detections, image)
    #         return av.VideoFrame.from_ndarray(RGB_img, format="bgr24")

    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>',
        unsafe_allow_html=True)
    activities = ["Home","Dashboard","Image", "Webcam"]
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox(
        "Choose among the given options:", activities)

    if choice == "Home":
        
  
        st.write("")
   
        st.title('A Face Mask Detection System')
        st.subheader('Built with OpenCV and Keras/TensorFlow leveraging Deep Learning and Computer Vision Concepts to detect face mask in still images as well as in real-time webcam streaming.')
        st.write('You can choose the options from the left.')
        st.write("")
    
        st.write("")
    
    
        st.write("")
    
        st.header('Upcoming Features: ')
        st.markdown("- Webcam Mask Detection")
        st.markdown("- Detecting Incorrect Mask")
    
    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', 
            unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=["jpg","jpeg","png"])  # upload image
        if image_file is not None:
            our_image = Image.open(image_file)  # making compatible to PIL
            im = our_image.save('./images/out.jpg')
            saved_image = st.image(
                image_file, caption='', use_column_width=True)
            st.markdown(
                '<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
            if st.button('Process'):
                # load the input image from disk and grab the image spatial
                # dimensions
                st.image(RGB_img, use_column_width=True)

    if choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        # webrtc_streamer(key="example")
        st.markdown("This feature will be available soon...")
    # run = st.checkbox('Open Webcam')
    # FRAME_WINDOW = st.image([])
    # camera = cv2.VideoCapture(0)
    # while run:
    #     # Reading image from video stream
    #     _, img = camera.read()
    #     # Call method we defined above
    #     # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #     img = predict(img)
    #       # st.image(img, use_column_width=True)
    #     FRAME_WINDOW.image(img)
    # if not run:
    #     st.write('Webcam has stopped.')


    if choice == 'Dashboard':
        covid_data_menu()
        

if __name__ == "__main__":
    
    mask_detection()