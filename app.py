import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 업로드된 파일을 저장할 경로 설정
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 이미지 처리 함수
def process_image(image_path):
    # 1. 이미지 읽기
    image = cv2.imread(image_path)

    # 2. 이미지 확대 (2.5배)
    scale_percent = 250  # 250% 확대
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    # 3. 그레이스케일로 변환 (흑백)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 4. Median Blur를 적용해 노이즈 감소 (커널 크기: 5로 적용)
    median_blurred_image = cv2.medianBlur(gray_image, 5)

    # 5. Bilateral Filter를 적용해 경계를 보존하며 노이즈 제거
    bilateral_filtered_image = cv2.bilateralFilter(median_blurred_image, 9, 75, 75)

    # 6. Gaussian Blur 추가 적용으로 더 많은 잡음 제거 (커널 크기: 3x3)
    blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (3, 3), 0)

    # 7. 이미지 샤프닝 (선명하게)
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel_sharpening)

    # 8. Adaptive Thresholding 적용 (동적으로 임계값 조정)
    adaptive_thresh = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 5)

    # 9. 모폴로지 열기 연산을 통해 잡음 제거 (침식 후 팽창)
    kernel = np.ones((1, 1), np.uint8)
    opened_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    # 10. 경계선 강화 (Canny Edge Detection)
    edges = cv2.Canny(opened_image, 100, 200)

    # 11. 허프 변환을 이용해 선 탐지
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=15)

    # 12. 탐지된 선을 Adaptive Thresholding 이미지에 새로 그림 (선 연결)
    processed_image_with_lines = opened_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(processed_image_with_lines, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 결과 이미지로 색상 반전
    white_background = np.ones_like(image) * 255
    white_background[processed_image_with_lines == 0] = [0, 0, 0]
    inverted_image = cv2.bitwise_not(white_background)

    # 결과를 파일로 저장
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.png')
    cv2.imwrite(output_path, inverted_image)

    return output_path


# 파일 업로드 및 처리 엔드포인트
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 이미지 처리
        processed_image_path = process_image(file_path)

        # 처리된 이미지 반환
        return send_file(processed_image_path, mimetype='image/png')


# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
