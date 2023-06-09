# Age recognition with webcam

This project uses [pre-trained](https://github.com/ximader/Portfolio/tree/main/yandex.practicum/ds_19_image_recognition) TensorFlow model to predict person's age by webcam video stream.


<img src="preview.png" alt="drawing" style="width:600px;"/>

 

## Libraries used:
- OpenCV
- Tensorflow
- Streamlit
- Numpy
- Pandas

## Hosted on Streamlit Cloud: 

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://age-recognition-webcam.streamlit.app/) https://age-recognition-webcam.streamlit.app/

## To run this demo locally:
```
conda create -n test_env python=3.9
conda activate test_env
pip install --upgrade streamlit opencv-python av tensorflow-macos streamlit_webrtc

streamlit run https://raw.githubusercontent.com/ximader/Portfolio/main/pet_projects/webcam_age_recognition_with_streamlit/streamlit_age_recognition_webcam.py
```
