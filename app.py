from flask import Flask, request, jsonify
import easyocr
import face_recognition
import base64
import cv2
import numpy as np
import os

app = Flask(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['ar', 'en'], gpu=False)

@app.route('/extract-id', methods=['POST'])
def extract_id():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # Decode Base64 image
        image_base64 = data['image']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        image_bytes = base64.b64decode(image_base64)

        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Temporarily save image to disk
        temp_path = 'temp_id.jpg'
        cv2.imwrite(temp_path, image)

        # OCR processing
        bounds = reader.readtext(
            temp_path,
            slope_ths=2,
            height_ths=2.0,
            width_ths=2.0
        )
        os.remove(temp_path)

        # Extract text list
        datalist = [entry[1] for entry in bounds]

        # Initialize fields
        id_number = ""
        name = ""
        lastname = ""
        birth_date = ""

        if len(datalist) >= 7:
            id_number = datalist[2]
            lastname = " ".join(datalist[3].split(" ")[1:])
            name = " ".join(datalist[4].split(" ")[1:])
            birth_date = " ".join(datalist[6].split(" ")[2:])

        result = {
            'id_number': id_number,
            'name': name,
            'lastname': lastname,
            'birth_date': birth_date
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        if not data or 'livePhoto' not in data or 'idImage' not in data:
            return jsonify({'error': 'Missing livePhoto or idImage'}), 400

        # Decode Base64 images
        live_photo_base64 = data['livePhoto']
        id_image_base64 = data['idImage']

        if ',' in live_photo_base64:
            live_photo_base64 = live_photo_base64.split(',')[1]
        if ',' in id_image_base64:
            id_image_base64 = id_image_base64.split(',')[1]

        live_photo_bytes = base64.b64decode(live_photo_base64)
        id_image_bytes = base64.b64decode(id_image_base64)

        # Convert to OpenCV images
        live_nparr = np.frombuffer(live_photo_bytes, np.uint8)
        id_nparr = np.frombuffer(id_image_bytes, np.uint8)

        live_image = cv2.imdecode(live_nparr, cv2.IMREAD_COLOR)
        id_image = cv2.imdecode(id_nparr, cv2.IMREAD_COLOR)

        if live_image is None or id_image is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # Convert images to RGB (face_recognition uses RGB)
        rgb_live = cv2.cvtColor(live_image, cv2.COLOR_BGR2RGB)
        rgb_id = cv2.cvtColor(id_image, cv2.COLOR_BGR2RGB)

        # Get face encodings
        id_encodings = face_recognition.face_encodings(rgb_id)
        live_encodings = face_recognition.face_encodings(rgb_live)

        if not id_encodings:
            return jsonify({'error': 'No face found in ID image'}), 400
        if not live_encodings:
            return jsonify({'error': 'No face found in live image'}), 400

        # Compare faces
        match = face_recognition.compare_faces(
            [id_encodings[0]],
            live_encodings[0],
            tolerance=0.6  # Lower is more strict
        )[0]

        # Get face distance
        face_distance = face_recognition.face_distance(
            [id_encodings[0]],
            live_encodings[0]
        )[0]

        return jsonify({
            'match': bool(match),
            'distance': float(face_distance),
            'threshold': 0.6  # The tolerance value used
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
