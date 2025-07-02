import pickle
import pandas as pd
import numpy as np

# Load the pickle file
with open('student_embeddings.pkl', 'rb') as f:
    embeddings_dict = pickle.load(f)

# Compute the mean of each embedding vector
averages = {student_id: np.mean(embedding) for student_id, embedding in embeddings_dict.items()}

# Convert to DataFrame
df = pd.DataFrame(list(averages.items()), columns=['student_id', 'embedding_mean'])

# Save to CSV
df.to_csv('student_embedding_means.csv', index=False)

print("Saved student means to student_embedding_means.csv") 