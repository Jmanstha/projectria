from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import onnxruntime as ort
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Secret key for sessions
app.secret_key = 'your_secret_key_here'

# File paths
CANDIDATES_FILE = 'candidates.txt'
VOTES_FILE = 'votes.txt'
VOTED_USERS_FILE = 'voted_users.txt'
USERS_FILE = 'users.txt'

# Load ONNX model for face recognition
onnx_model_path = 'facenet.onnx'  # Path to your ONNX model
onnx_session = ort.InferenceSession(onnx_model_path)

# Function to preprocess the image for face recognition
def preprocess_image(image, target_size=(160, 160)):
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    image = (image - 127.5) / 128.0
    return image

# Function to extract face embeddings using ONNX model
def get_face_embedding(image, model_session):
    preprocessed_image = preprocess_image(image)
    input_name = model_session.get_inputs()[0].name
    embedding = model_session.run(None, {input_name: preprocessed_image})[0]
    return embedding

# Function to compare two embeddings using cosine similarity
def compare_faces(embedding1, embedding2):
    return cosine_similarity([embedding1.flatten()], [embedding2.flatten()])[0][0]

# Function to detect faces using OpenCV
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return image[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]] if len(faces) > 0 else None

# Function to decode base64 image
# Function to decode base64 image
def decode_base64_image(data):
    img = Image.open(BytesIO(base64.b64decode(data.split(',')[1])))
    # Convert the image to RGB mode to avoid the RGBA error
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    return img


# Function to load candidates
def load_candidates():
    candidates = []
    if os.path.exists(CANDIDATES_FILE):
        with open(CANDIDATES_FILE, 'r') as file:
            for line in file:
                name, image = line.strip().split('|')
                candidates.append({'name': name, 'image': image})
    return candidates

# Function to load votes
def load_votes():
    votes = {}
    if os.path.exists(VOTES_FILE):
        with open(VOTES_FILE, 'r') as file:
            for line in file:
                name, vote_count = line.strip().split('|')
                votes[name] = int(vote_count)
    return votes

# Function to save votes
def save_votes(votes):
    with open(VOTES_FILE, 'w') as file:
        for name, vote_count in votes.items():
            file.write(f"{name}|{vote_count}\n")

# Function to load voted users
def load_voted_users():
    voted_users = set()
    if os.path.exists(VOTED_USERS_FILE):
        with open(VOTED_USERS_FILE, 'r') as file:
            for line in file:
                voted_users.add(line.strip())
    return voted_users

# Function to save voted user
def save_voted_user(user_id):
    with open(VOTED_USERS_FILE, 'a') as file:
        file.write(f"{user_id}\n")

# Function to check if user exists
def user_exists(user_id):
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as file:
            for line in file:
                if line.strip() == user_id:
                    return True
    return False

@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Check if the user has voted
    voted_users = load_voted_users()
    user_id = session['user_id']

    if user_id in voted_users:
        return redirect(url_for('thank_you'))

    candidates = load_candidates()
    votes = load_votes()

    for candidate in candidates:
        candidate['votes'] = votes.get(candidate['name'], 0)

    return render_template('index.html', candidates=candidates)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['user_id']

        if not user_exists(user_id):
            flash("User ID does not exist.", "error")
            return redirect(url_for('login'))

        session['user_id'] = user_id

        voted_users = load_voted_users()
        if user_id in voted_users:
            return redirect(url_for('thank_you'))

        return redirect(url_for('index'))

    return render_template('login.html')

@app.route('/vote/<candidate_name>', methods=['POST'])
def vote(candidate_name):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    voted_users = load_voted_users()

    if user_id in voted_users:
        return redirect(url_for('thank_you'))

    votes = load_votes()
    if candidate_name in votes:
        votes[candidate_name] += 1
    else:
        votes[candidate_name] = 1

    save_votes(votes)
    save_voted_user(user_id)

    return redirect(url_for('thank_you'))

@app.route('/capture_image', methods=['POST'])
@app.route('/capture_image', methods=['POST'])
def capture_image():
    data = request.get_json()  # Get JSON data from frontend
    image_data = data.get('image')  # Extract the 'image' from the JSON data

    if not image_data:
        return jsonify({"status": "error", "message": "No image data received"}), 400

    img = decode_base64_image(image_data)
    user_id = session.get('user_id', 'unknown')
    img_path = os.path.join('static', 'userimage', f'{user_id}.jpg')
    img.save(img_path)

    reference_image = cv2.imread(img_path)
    detected_face = detect_face(reference_image)

    if detected_face is None:
        return jsonify({"status": "error", "message": "No face detected"}), 400

    ref_embedding = get_face_embedding(detected_face, onnx_session)
    
    # Process captured image for comparison
    test_face = detect_face(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
    if test_face is None:
        return jsonify({"status": "error", "message": "No face detected in captured image"}), 400
    
    test_embedding = get_face_embedding(test_face, onnx_session)
    similarity_score = compare_faces(ref_embedding, test_embedding)

    if similarity_score > 0.7:
        return jsonify({"status": "success", "message": "Face matched!", "redirect": True})
    
    else:
        return jsonify({"status": "error", "message": "Face does not match!", "redirect": False})



@app.route('/thank_you')
def thank_you():
    return render_template('thankyou.html')

if __name__ == '__main__':
    app.run(debug=True)
