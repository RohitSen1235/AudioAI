import streamlit as st
import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
# Utility function

# Step 1: Preprocessing and feature extraction
def extract_features(audio_path, num_mfcc=16):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    # spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    # spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    # spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    # spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features = np.concatenate(
        # (mfcc, spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff),
        (mfcc,),
        axis=0
    )
    return features

# Step 2: Build a reference library
def build_reference_library(list_of_files):
    reference_library = {}
    for file in list_of_files:
        features = extract_features(file)
        reference_library[file.name] = features
    return reference_library


# Step 3: Compare query audio file with reference library
def compare_audio(query_audio, reference_library):
    query_features = extract_features(query_audio)
    similarities = {}
    for filename, reference_features in reference_library.items():
        # minkowski_distance = distance.minkowski(np.array(query_features.T), np.array(reference_features.T))
        # similarity = 1 / (1+minkowski_distance)
        similarity = cosine_similarity(query_features.T, reference_features.T).mean()
        similarities[filename] = similarity
    return similarities,query_features

# Check if a given song exists in the reference library
def is_song_present(query_audio, reference_library, threshold=0.8):
    similarities, query_features = compare_audio(query_audio, reference_library)

    # Sort the similarities dictionary by values (descending order)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    distance={}
    # Print the most similar audio files
    for filename, similarity in sorted_similarities:
        distance[filename] =  np.linalg.norm(reference_library[filename] - query_features)
        print(f"{filename}: Similarity : {similarity}, distance : {distance[filename]}")

    min_distance = min(distance.values())
    if min_distance == 0.0:
        print(f"minimum distance : {min_distance}")
        return True
    # max_similarity = max(similarities.values())
    # if max_similarity >= threshold:

    else:
        return False
    
def main():
    # Set page title and icon
    st.set_page_config(page_title="AudioAI", page_icon="🎵")

    # Add header strip
    st.title("AudioAI")

    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # File upload section
        library = st.file_uploader("Upload Library", type=['WAV', 'MP3'], key="Upload Song Library" ,accept_multiple_files=True)

    
    with col2:

        file = st.file_uploader("Upload Query Song", type= ['WAV', 'MP3'], key="Upload Song")

    if file and library:
        if st.button("Check for song in library"):
            library_embeddings = {}
            with st.spinner('Generating Vector Embeddings for Songs in Library'):
                # st.write("Generating Vector Embeddings for Songs in Library")
                library_embeddings = build_reference_library(library)
            
            with st.spinner('Comparing Query Song with Library'):
                song_exists = is_song_present(file, library_embeddings)
                if song_exists:
                    st.warning(f'song {file.name} PRESENT in library', icon="⚠️")
                else:
                    st.success(f"The song {file.name} is not present in the given library.", icon="✅")

if __name__ == "__main__":
    main()

