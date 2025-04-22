import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import torch
import tempfile
import cv2
from ultralytics import YOLO

# Load the saved model
model = tf.keras.models.load_model("traffic_sign_model.h5")

# --- Load YOLOv5 Model ---
#yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
yolo_model = YOLO('best.pt')

# --- Class Labels ---
class_names = [
    "Speed Limit 20", "Speed Limit 30", "Speed Limit 50", "Speed Limit 60",
    "Speed Limit 70", "Speed Limit 80", "End of Speed Limit 80",
    "Speed Limit 100", "Speed Limit 120", "No Passing", "No Passing for trucks",
    "Right of way at intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 tons prohibited", "No entry",
    "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on right",
    "Road work", "Traffic signals", "Pedestrians sign", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles > 3.5 tons", "Pedestrian crossing", "Traffic signals"
]

# --- Streamlit UI ---
st.set_page_config(page_title="🚦 Traffic Sign + YOLO Detection", layout="centered")
st.title("🚦 Traffic Sign Recognition + YOLOv5 Detection")
st.write("Upload an image, and the CNN will classify the traffic sign. If it's 'Traffic signals', YOLOv5 will detect objects.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    # --- CNN Prediction ---
    image = Image.open(uploaded_file)  # PIL image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = image.convert("RGB")       # ensure 3-channel RGB
    image_np = np.array(image)         # convert to numpy array
    img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0  # Normalize
    test_img = np.expand_dims(img, axis=0)
    prediction = model.predict(test_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"🔍 Predicted Sign: **{class_names[predicted_class]}**")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # --- If Class == 44 ("Traffic signals"), Run YOLOv5 ---
    if predicted_class == 43:
        st.markdown("---")
        st.subheader("🧠 YOLOv5 Object Detection (Traffic Signal Detected!)")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp_path = temp.name
            image.save(temp_path)

        # Run YOLO
        yolo_results = yolo_model(temp_path)
        yolo_results.render()  # modifies yolo_results.ims[0]

        result_img = Image.fromarray(yolo_results.ims[0])
        st.image(result_img, caption="YOLOv5 Detection Result", use_column_width=True)

        # Show detection results
        detections = yolo_results.pandas().xyxy[0]
        if not detections.empty:
            st.write("📋 Detected Objects:")
            st.dataframe(detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']])
        else:
            st.info("No objects detected.")
