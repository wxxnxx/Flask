from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS  # CORS 추가
import cv2
import numpy as np
import os
import logging

app = Flask(__name__)
CORS(app)  # CORS 설정

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)


# 이미지 처리 함수
def process_image(image_path, original_filename):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {image_path}")

    # 이미지 읽기
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"유효하지 않은 이미지 파일입니다: {image_path}")

    # 입력된 이미지 크기에 따른 확대/축소 비율 결정
    max_dimension = max(image.shape[0], image.shape[1])
    scale_percent = 130 if max_dimension < 1800 else 50  # 2배 확대 또는 50% 축소

    # 이미지 크기 변경
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

    # 그레이스케일 변환 및 필터링
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median_blurred_image = cv2.medianBlur(gray_image, 3)
    bilateral_filtered_image = cv2.bilateralFilter(median_blurred_image, 9, 75, 75)
    blurred_image = cv2.GaussianBlur(bilateral_filtered_image, (3, 3), 0)

    # 샤프닝 필터 적용
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    sharpened_image = cv2.filter2D(blurred_image, -1, kernel_sharpening)

    # Adaptive Thresholding 적용
    adaptive_thresh = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 25, 5)

    # 모폴로지 열기 연산
    kernel = np.ones((2, 2), np.uint8)
    opened_image = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)

    # Canny Edge Detection 적용
    edges = cv2.Canny(opened_image, 100, 200)

    # 허프 변환으로 선 탐지
    lines = cv2.HoughLinesP(edges, 2, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=5)

    # 탐지된 선 그리기
    processed_image_with_lines = opened_image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(processed_image_with_lines, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # 최종 결과 생성
    white_background = np.ones_like(image) * 255
    white_background[processed_image_with_lines == 0] = [0, 0, 0]

    # 색상 반전
    inverted_image = cv2.bitwise_not(white_background)

    # 결과 이미지 저장 (원본 파일 이름 사용)
    base, ext = os.path.splitext(original_filename)
    result_path = os.path.join('uploads', f'{base}_processed{ext}')
    success = cv2.imwrite(result_path, inverted_image)

    if not success:
        raise IOError(f"결과 이미지 저장 실패: {result_path}")

    logging.debug(f"결과 이미지 저장 경로: {result_path}")
    return result_path


@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '선택된 파일이 없습니다.'}), 400

    # uploads 디렉토리 경로 설정
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)  # 디렉토리가 없으면 생성

    # 파일 저장 경로
    image_path = os.path.join(uploads_dir, file.filename)
    file.save(image_path)
    logging.debug(f"파일 저장 경로: {image_path}")

    try:
        # 이미지 처리
        result_path = process_image(image_path, file.filename)
    except Exception as e:
        logging.error(f"이미지 처리 오류: {e}")
        return jsonify({'error': str(e)}), 500

    # 결과 이미지 반환
    return send_file(result_path, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
