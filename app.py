import streamlit as st
import cv2
import numpy as np

st.set_page_config(
    page_title="Road Lane Lines Detection",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ Road Lane Lines Detection")
st.write("Upload a road image to detect lane lines using OpenCV.")

uploaded_file = st.file_uploader(
    "Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = edges.shape
    mask = np.zeros_like(edges)

    polygon = np.array([[
        (int(width * 0.10), height),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=20,
        maxLineGap=300
    )

    line_image = np.zeros_like(image)

    if lines is not None:
       for line in lines:
    coordinates = np.asarray(line).reshape(-1)

    if coordinates.size < 4:
        continue

    x1, y1, x2, y2 = map(int, coordinates[:4])

    cv2.line(
        line_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        10
    )

    result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

    return edges, masked_edges, result, lines

if uploaded_file is not None:
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error("Unable to read this image. Please upload another image.")
    else:
        edges, masked_edges, result, lines = detect_lanes(image)

        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Road Image")
            st.image(original_rgb, use_container_width=True)

        with col2:
            st.subheader("Detected Lane Lines")
            st.image(result_rgb, use_container_width=True)

        st.subheader("Canny Edge Detection")
        st.image(edges, clamp=True, use_container_width=True)

        st.subheader("Region of Interest")
        st.image(masked_edges, clamp=True, use_container_width=True)

        if lines is not None:
            st.success(f"Detection completed! {len(lines)} line segments detected.")
        else:
            st.warning("No lane lines detected. Try another road image.")

        st.download_button(
            "Download Detected Image",
            data=cv2.imencode(
                ".jpg", result
            )[1].tobytes(),
            file_name="lane_detection_output.jpg",
            mime="image/jpeg"
        )
else:
    st.info("Please upload a road image to start lane detection.")

st.caption("Built using Python, OpenCV and Streamlit.")
