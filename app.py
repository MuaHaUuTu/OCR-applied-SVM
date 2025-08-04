import streamlit as st
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np
import subprocess
import os


def run_inference(image_path, config_path, weight_path):
    """
    Chạy lệnh inference OCR từ paddleocr thông qua subprocess.
    """
    command = [
        "python", "PaddleOCR/tools/infer_rec.py",
        "-c", config_path, 
        "-o", f"Global.pretrained_model={weight_path}",
        f"Global.infer_img={image_path}"
    ]
    
    # Chạy lệnh trong subprocess
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        return stdout.decode(), None  # Trả về kết quả stdout nếu thành công
    else:
        return None, stderr.decode('utf-8', errors='ignore')  # Bỏ qua các lỗi mã hóa




# Khởi tạo mô hình OCR
ocr = PaddleOCR(use_angle_cls=True, lang='vi')  # Cấu hình ngôn ngữ phù hợp (ví dụ: 'vi' cho tiếng Việt)

# Tiêu đề ứng dụng
st.title("SVTR Text Recognition")

# Upload file ảnh
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Đọc ảnh bằng PIL
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Lưu ảnh vào thư mục tạm thời
    image_path = 'uploaded_image.png'
    image.save(image_path)

    # Đường dẫn tới mô hình và tệp cấu hình
    config_path = 'rec_svtr_large_10local_11global_stn_en.yml'  # Cập nhật đường dẫn chính xác
    weight_path = 'weight_svtr.pdparams'  # Cập nhật đường dẫn chính xác

    # Chạy inference và nhận kết quả
    output, error = run_inference(image_path, config_path, weight_path)

    if error:
        st.error(f"Error during inference: {error}")
    else:
        # Chuyển kết quả ra thành các bounding boxes và text
        results = output.splitlines()
        boxes = []
        texts = []
        scores = []

        # Chuyển đổi kết quả từ output thành list của bounding boxes, text, và scores
        for result in results:
            parts = result.split('\t')
            if len(parts) >= 2:
                
                text = parts[1]
                score = float(parts[2]) if len(parts) > 2 else 0.0
                
                texts.append(text)
                scores.append(score)

        # Vẽ kết quả OCR lên ảnh
        img_array = np.array(image)
        

        # Hiển thị kết quả nhận dạng văn bản
        st.subheader("Recognized Text:")
        for text in texts:
             st.markdown(f"<p style='font-size:20px; font-weight:bold;'>{text}</p>", unsafe_allow_html=True)
 # Hiển thị text nhận dạng được

       

# Footer
st.write("Developed using Streamlit and PaddleOCR")
