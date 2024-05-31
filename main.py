import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io

buffer = io.BytesIO()
buffer2 = io.BytesIO()

model = YOLO("best.pt")

def prediction(image, conf):
    result = model.predict(image, conf = conf)
    boxes = result[0].boxes
    res_plotted = result[0].plot()[:, :, ::-1]
    return res_plotted

st.title('Deteksi Penyakit Pada Daun Mangga')

values = st.slider(
    label='Pilih Confidence',
    value=(1.0))
st.write('Confidence', values)

image = st.camera_input('Take a picture')
if image:
    image = Image.open(image)
    pred = prediction(image, values)
    st.image(pred)
    
    im = Image.fromarray(pred)
    im.save(buffer, format="PNG")
    
    st.download_button(
        key = 1,
        label="Download",
        data= buffer,
        file_name="Deteksi Penyakit.png",
        mime="image/png",
    )

image2 = st.file_uploader('Upload', type = ['jpg', 'jpeg', 'png'])
if image2:
    image2 = Image.open(image2)
    pred = prediction(image2, values)
    st.image(pred)
    
    im = Image.fromarray(pred)
    im.save(buffer2, format="PNG")
    
    st.download_button(
        key = 2,
        label="Download",
        data= buffer2,
        file_name="Deteksi Penyakit.png",
        mime="image/png",
    )
    
