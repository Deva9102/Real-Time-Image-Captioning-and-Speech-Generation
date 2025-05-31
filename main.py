import streamlit as st

from PIL import Image

import numpy as np

import os

import pickle

import numpy as np

from tqdm.notebook import tqdm

import cv2

import json

from keras.models import load_model

import pickle

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import collections

import base64

from keras.models import load_model

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from keras.models import Model

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model

from tensorflow.keras.utils import to_categorical, plot_model

from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add

from gtts import gTTS

from googletrans import Translator

 

st.set_page_config(

    initial_sidebar_state="expanded",

    page_title="VizCap"

)  

hide = """

        <style>

            #MainMenu {visibility: hidden;}

            footer {visibility: hidden;}

        </style>

    """

st.markdown(hide, unsafe_allow_html=True)




if __name__ == '__main__':

    # Path to your GIF file

   

    gif_path = "/Volumes/One Touch/GAIP JULY/ezgif.com-resize.gif"

    # Read the GIF file

    with open(gif_path, "rb") as f:

        contents = f.read()

    data_url = base64.b64encode(contents).decode("utf-8")

    st.markdown(

    f"""

    <div style="display:flex; justify-content:center;">

        <img src="data:image/gif;base64,{data_url}" alt="GIF" >

    </div>

    """,

     unsafe_allow_html=True,)




    st.text("")

    st.text("")

    st.markdown(

    """

    <style>

    /* Style the title for st.camera_input */

    .stCameraFrame span {

        display: block;

        font-size: 500px;

        font-weight: bold;

        text-align: center;

        margin-bottom: 10px;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

    st.title(" :movie_camera: VizCap:studio_microphone:")

    st.text("Steps to use : ")

    st.text("1.Click on take photo to capture an image\n2.Then,generate caption\n3.Select desired audio language and click on convert\n4.Play the audio")

    st.text("Tips : For best results, click image in a well lit environment")

 

   

    st.markdown('''### Webcam  :movie_camera: ''')

    img_file_buffer = st.camera_input("")

 

    if img_file_buffer is not None:

        # To read image file buffer as a PIL Image:

        image = Image.open(img_file_buffer)

        st.image(image)

        # reshape data for model

        #image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # Check the type of img_array:

        # Should output: <class 'numpy.ndarray'>

        #st.write(type(image))

 

        # Check the shape of img_array:

        # Should output shape: (height, width, channels)

        #st.write(image.shape) #streamlit run camera2.py

 

    with st.sidebar:

        st.sidebar.markdown(

            """<style>

            .big-font {

            font-size:30px !important;}

            </style>""", unsafe_allow_html=True)

        st.sidebar.markdown('<p class="big-font">About us !!</p>', unsafe_allow_html=True)

        st.sidebar.markdown('<p> Sight is a function of the eyes, but vision is a function of the heart - Myles Munroe</p>', unsafe_allow_html=True)

        st.sidebar.markdown('<p> VizCap is an innovative application for real-time image captioning and text-to-speech conversion. It clicks a picture in real-time and gives out a caption which is then converted into speech in multiple languages.Our goal is to help people with visual impairment comprehend their surroundings better and connect with their environment by using our application.</p>', unsafe_allow_html=True)

        st.sidebar.markdown(' GAIP NUS 2023 July TEAM, NUMBER 6 ')

        st.sidebar.markdown('Ligandro S.Y.,P.T. Devadarshini,Bhavya Sree Pyla,Soham Chakrabarti,M.Sherlin Jenifer')

 

    st.markdown(

    """

    <style>

    /* Import the Acme font */

    @import url('https://fonts.googleapis.com/css2?family=poppins&display=swap');

 

    /* Apply Acme font to the text */

    .streamlit-text-container {

        font-family: 'Acme', sans-serif;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

    streamlit_style = """

        <style>

        @import url('https //fonts.googleapis.com/css family=poppins&display=swap');

        html, body, [class*="css"]  {

        font-family: 'Acme', sans-serif;

        }

        </style>

        """

 

    st.markdown(streamlit_style, unsafe_allow_html=True)

       

    if 'result' not in st.session_state:

        st.session_state.result = None

       

    caption_generated = ""

    # img_bytes earlier

    if st.button('Generate captions!'):

   

        # Read the files word_to_idx.pkl and idx_to_word.pkl to get the mappings between word and index

        word_to_index = {}

        with open ("/Volumes/One Touch/Liga_30K/word_to_idx.pkl", 'rb') as file:

            word_to_index = pd.read_pickle(file, compression=None)

 

        index_to_word = {}

        with open ('/Volumes/One Touch/Liga_30K/idx_to_word.pkl', 'rb') as file:

            index_to_word = pd.read_pickle(file, compression=None)




        print("Loading the model...")

        model = load_model('/Volumes/One Touch/Liga_30K/liga_30Kv2.h5')

 

        resnet50_model = ResNet50 (weights = 'imagenet', input_shape = (224, 224, 3))

        resnet50_model = Model (resnet50_model.input, resnet50_model.layers[-2].output)




        # Generate Captions for a random image

        # Using Greedy Search Algorithm

 

        def predict_caption(photo):

 

            inp_text = "startseq"

 

            #max_len = 80 which is maximum length of caption

            for i in range(80):

                sequence = [word_to_index[w] for w in inp_text.split() if w in word_to_index]

                sequence = pad_sequences([sequence], maxlen=80, padding='post')

 

                ypred = model.predict([photo, sequence])

                ypred = ypred.argmax()

                word = index_to_word[ypred]

 

                inp_text += (' ' + word)

 

                if word == 'endseq':

                    break

 

            final_caption = inp_text.split()[1:-1]

            final_caption = ' '.join(final_caption)

            return final_caption




        def preprocess_image (img):

            # Reshape the image to 224x224

            img = img.resize((224, 224))

            img = img_to_array(img)

            # Convert 3D tensor to a 4D tendor

            img = np.expand_dims(img, axis=0)

 

            #Normalize image accoring to ResNet50 requirement

            img = preprocess_input(img)

 

            return img

 

        # A wrapper function, which inputs an image and returns its encoding (feature vector)

        def encode_image (img):

            img = preprocess_image(img)

 

            feature_vector = resnet50_model.predict(img)

            # feature_vector = feature_vector.reshape((-1,))

            return feature_vector

 

        def runModel(img_name):

            #img_name = input("enter the image name to generate:\t")

 

            print("Encoding the image ...")

            photo = encode_image(img_name).reshape((1, 2048))




            print("Running model to generate the caption...")

            caption = predict_caption(photo)

 

            #plt.show()

            print(caption)

            return caption

       

        # predict from the trained model

        try:

            caption_generated = runModel(image)

            #caption_generated = caption_generated[9:]

            #caption_generated = caption_generated[:-6]

        finally:

            st.session_state.result = caption_generated

        st.write(f" {caption_generated}")

 

       

    try:

        os.mkdir("temp")

    except:

        pass

    st.markdown('''### Text to speech:studio_microphone:''')

    text = st.session_state.result

    translator = Translator()

    in_lang = st.selectbox(

        "Select your input language",

        ("English", "Hindi", "Bengali", "korean", "Chinese", "Japanese"),

    )

    if in_lang == "English":

        input_language = "en"

    elif in_lang == "Hindi":

        input_language = "hi"

    elif in_lang == "Bengali":

        input_language = "bn"

    elif in_lang == "korean":

        input_language = "ko"

    elif in_lang == "Chinese":

        input_language = "zh-cn"

    elif in_lang == "Japanese":

        input_language = "ja"

 

    out_lang = st.selectbox(

        "Select your output language",

        ("English", "Hindi", "Bengali", "korean", "Chinese", "Japanese"),

    )

    if out_lang == "English":

        output_language = "en"

    elif out_lang == "Hindi":

        output_language = "hi"

    elif out_lang == "Bengali":

        output_language = "bn"

    elif out_lang == "korean":

        output_language = "ko"

    elif out_lang == "Chinese":

        output_language = "zh-cn"

    elif out_lang == "Japanese":

        output_language = "ja"

 

    english_accent = st.selectbox(

        "Select your english accent",

        (

            "Default",

            "India",

            "United Kingdom",

            "United States",

            "Canada",

            "Australia",

            "Ireland",

            "South Africa",

        ),

    )

 

    if english_accent == "Default":

        tld = "com"

    elif english_accent == "India":

        tld = "co.in"

 

    elif english_accent == "United Kingdom":

        tld = "co.uk"

    elif english_accent == "United States":

        tld = "com"

    elif english_accent == "Canada":

        tld = "ca"

    elif english_accent == "Australia":

        tld = "com.au"

    elif english_accent == "Ireland":

        tld = "ie"

    elif english_accent == "South Africa":

        tld = "co.za"

 

    def text_to_speech(input_language, output_language, text, tld):

        translation = translator.translate(text, src=input_language, dest=output_language)

        trans_text = translation.text

        tts = gTTS(trans_text, lang=output_language, tld=tld, slow=False)

        try:

            my_file_name = text[0:20]

        except:

            my_file_name = "audio"

        tts.save(f"temp/{my_file_name}.mp3")

        return my_file_name, trans_text

 

    display_output_text = st.checkbox("Display output text")

 

    if st.button("convert"):

        result, output_text = text_to_speech(input_language, output_language, text, tld)

        audio_file = open(f"temp/{result}.mp3", "rb")

        audio_bytes = audio_file.read()

        st.markdown(f"## Your audio:")

        st.audio(audio_bytes, format="audio/mp3", start_time=0)

 

        if display_output_text:

            st.markdown(f"## Output text:")

            st.write(f" {output_text}")

 

   

    def remove_files(n):

        mp3_files = glob.glob("temp/*mp3")

        if len(mp3_files) != 0:

            now = time.time()

            n_days = n * 86400

            for f in mp3_files:

                if os.stat(f).st_mtime < now - n_days:

                    os.remove(f)

                    print("Deleted ", f)

                   

st.markdown(

    """

    <style>

    .stCameraFrame span {

        display: block;

        font-size: 500px;

        font-weight: bold;

        text-align: center;

        margin-bottom: 10px;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

st.markdown("### Rate your experience")

rating = st.slider("", min_value=0, max_value=5, step=1)

st.write("You rated:", rating, "out of 5")

 

import streamlit as st

 

if 'like_count' not in st.session_state:

    st.session_state.like_count = 0

 

st.markdown(

    """

    <style>

    .footer {

        position: fixed;

        left: 0;

        bottom: 0;

        width: 100%;

        background-color: #282829;

        padding: 10px;

        text-align: right;

        font-size: 20px;

        color: #ffffff;

    }

    .footer .like_count {

        margin-right: 30px;

        color: #ffffff;

    }

    </style>

    """,

    unsafe_allow_html=True,

)

 

javascript = """

<script>

function updateLikes() {

    var like_count = document.getElementById('like_count');

    var currentCount = parseInt(like_count.innerText);

    like_count.innerText = currentCount + 1;

}

</script>

"""

 

st.markdown(javascript, unsafe_allow_html=True)

 

if 'result' not in st.session_state:

    st.session_state.result = None

 

if st.button("Like &#10084;&#65039;", key="like_button", help="Like this page"):

    if 'like_count' not in st.session_state:

        st.session_state.like_count = 1

    else:

        st.session_state.like_count += 1

 

st.write(f"Liked by you and {st.session_state.like_count} others")

 

st.markdown(

    """

    <div class="footer">

        2023 VizCap

    </div>

    """.format(st.session_state.like_count),

    unsafe_allow_html=True,

)
