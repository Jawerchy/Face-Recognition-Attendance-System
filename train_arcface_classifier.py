import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from insightface.app import FaceAnalysis

# --- Configuration ---
DATASET_DIR = 'dataset'
CLASSIFIER_PATH = 'arcface_classifier.joblib'
LABELS_PATH = 'arcface_labels.joblib'
EMBEDDINGS_PATH = 'arcface_embeddings.npy'
LABELS_NUMPY_PATH = 'arcface_labels.npy'

# --- Step 1: Load Dataset ---
def load_dataset(dataset_dir):
    image_paths = []
    labels = []
    for student_id in os.listdir(dataset_dir):
        student_folder = os.path.join(dataset_dir, student_id)
        if os.path.isdir(student_folder):
            for fname in os.listdir(student_folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(student_folder, fname))
                    labels.append(student_id)
    return image_paths, labels

# --- Step 2: Extract ArcFace Embeddings ---
def extract_embeddings(image_paths, face_analyzer):
    embeddings = []
    for img_path in tqdm(image_paths, desc='Extracting embeddings'):
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        faces = face_analyzer.get(img_np)
        if faces:
            # Use the largest face (if multiple)
            face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])
            embeddings.append(face.embedding)
        else:
            print(f'No face found in {img_path}, skipping.')
            embeddings.append(np.zeros(512))  # ArcFace default embedding size
    return np.array(embeddings)

if __name__ == '__main__':
    print('Loading dataset...')
    image_paths, labels = load_dataset(DATASET_DIR)
    print(f'Total images: {len(image_paths)}')

    print('Initializing ArcFace...')
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(224, 224))

    print('Extracting embeddings...')
    embeddings = extract_embeddings(image_paths, face_analyzer)
    np.save(EMBEDDINGS_PATH, embeddings)
    np.save(LABELS_NUMPY_PATH, np.array(labels))

    print('Encoding labels...')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    joblib.dump(label_encoder, LABELS_PATH)

    print('Training classifier...')
    clf = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    clf.fit(embeddings, y)
    joblib.dump(clf, CLASSIFIER_PATH)

    print('Training complete!')
    print(f'Classifier saved to {CLASSIFIER_PATH}')
    print(f'Label encoder saved to {LABELS_PATH}') 