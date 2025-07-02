import numpy as np
import joblib
from PIL import Image
from insightface.app import FaceAnalysis

CLASSIFIER_PATH = 'arcface_classifier.joblib'
LABELS_PATH = 'arcface_labels.joblib'

# Load ArcFace model, classifier, and label encoder once
face_analyzer = None
classifier = None
label_encoder = None

def load_arcface_system():
    global face_analyzer, classifier, label_encoder
    if face_analyzer is None:
        face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_analyzer.prepare(ctx_id=0, det_size=(224, 224))
    if classifier is None:
        classifier = joblib.load(CLASSIFIER_PATH)
    if label_encoder is None:
        label_encoder = joblib.load(LABELS_PATH)


def recognize_student_arcface(pil_image):
    """
    Recognize a student from a PIL image using ArcFace and the trained classifier.
    Returns (student_id, confidence) or (None, None) if not recognized.
    """
    load_arcface_system()
    img = pil_image.convert('RGB')
    img_np = np.array(img)
    faces = face_analyzer.get(img_np)
    if not faces:
        return None, None
    # Use the largest face
    face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
    embedding = face.embedding.reshape(1, -1)
    probs = classifier.predict_proba(embedding)[0]
    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]
    student_id = label_encoder.inverse_transform([pred_idx])[0]
    # You can set a confidence threshold (e.g., 0.7)
    if confidence < 0.7:
        return None, confidence
    return student_id, confidence 