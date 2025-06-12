# import streamlit as st
# from PIL import Image
# import tempfile
# import os
# from ultralytics import YOLO

# # Page configuration
# st.set_page_config(page_title="🫀 Pacemaker Detection", layout="centered")

# st.markdown("""
#     <style>
#         .main {
#             background-color: #f0f2f6;
#         }
#         .block-container {
#             padding-top: 2rem;
#             padding-bottom: 2rem;
#         }
#         .stButton>button {
#             background-color: #ff4b4b;
#             color: white;
#             font-size: 16px;
#             border-radius: 8px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🫀 Heart Pacemaker Detection")
# st.markdown("""
# Upload a **heart X-ray image**, and this app will use a **YOLOv8 model** to detect whether a **pacemaker** is present in the image.
# """)

# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")  # Make sure this is correct

# model = load_model()

# uploaded_file = st.file_uploader("📤 Upload a Heart X-ray Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     # Save uploaded image to temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
#         image.save(temp_file.name)
#         temp_image_path = temp_file.name

#     st.subheader("🔎 Running Detection...")
#     results = model.predict(temp_image_path, conf=0.25, save=False)

#     # ✅ Use in-memory image from result
#     result_image = results[0].plot()  # Numpy array with drawn boxes
#     st.image(result_image, caption="📌 Detection Result", use_column_width=True)











### tetsing ##############



# import streamlit as st
# from PIL import Image
# import tempfile
# import os
# from ultralytics import YOLO
# import numpy as np
# import io

# # Page configuration
# st.set_page_config(page_title="🫀 Pacemaker Detection", layout="centered")

# st.markdown("""
#     <style>
#         .main {
#             background-color: #f0f2f6;
#         }
#         .block-container {
#             padding-top: 2rem;
#             padding-bottom: 2rem;
#         }
#         .stButton>button {
#             background-color: #ff4b4b;
#             color: white;
#             font-size: 16px;
#             border-radius: 8px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🫀 Heart Pacemaker Detection")
# st.markdown("""
# Upload a **heart X-ray image**, and this app will use a **YOLOv8 model** to detect whether a **pacemaker** is present in the image.
# """)

# @st.cache_resource
# def load_model():
#     return YOLO("best.pt")  # Make sure this file exists

# model = load_model()

# uploaded_file = st.file_uploader("📤 Upload a Heart X-ray Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

#     # Save uploaded image temporarily
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
#         image.save(temp_file.name)
#         temp_image_path = temp_file.name

#     st.subheader("🔎 Running Detection...")
#     results = model.predict(temp_image_path, conf=0.25, save=False)

#     # Display result image
#     result_image_np = results[0].plot()
#     st.image(result_image_np, caption="📌 Detection Result", use_column_width=True)

#     # Convert NumPy array to PIL Image for download
#     result_pil_image = Image.fromarray(result_image_np)
#     img_buffer = io.BytesIO()
#     result_pil_image.save(img_buffer, format="PNG")
#     img_buffer.seek(0)

#     # Download button
#     st.download_button(
#         label="⬇️ Download Result Image",
#         data=img_buffer,
#         file_name="pacemaker_detection_result.png",
#         mime="image/png"
#     )








############ condition 2 
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
from ultralytics import YOLO
import numpy as np
import io


st.set_page_config(page_title="🫀 Pacemaker Detection", layout="centered")

st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            font-size: 16px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🫀 Heart Pacemaker Detection")
st.markdown("""
Upload a **heart X-ray image**, and this app will use a **YOLOv8 model** to detect whether a **pacemaker** is present in the image.
""")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  

model = load_model()

uploaded_file = st.file_uploader("📤 Upload a Heart X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        temp_image_path = temp_file.name

    st.subheader("🔎 Running Detection...")
    results = model.predict(temp_image_path, conf=0.25, save=False)


    filtered_boxes = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > 0.7:
                filtered_boxes.append(box)

    if filtered_boxes:
        
        annotated_pil = image.copy()
        draw = ImageDraw.Draw(annotated_pil)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"pacemaker {conf:.2f}", fill="red", font=font)

        st.image(annotated_pil, caption="📌 Filtered Detection Result", use_column_width=True)

        # Download button
        img_buffer = io.BytesIO()
        annotated_pil.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        st.download_button(
            label="⬇️ Download Result Image",
            data=img_buffer,
            file_name="pacemaker_detection_result.png",
            mime="image/png"
        )
    else:
        st.warning("⚠️ No pacemaker detected with confidence > 70%.")
