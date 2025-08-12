Exploring Deep Learning Models for Music Genre Classification: Analysis of the GTZAN Dataset
A comprehensive study evaluating the effectiveness of Convolutional Neural Networks (CNNs) for music genre classification using the GTZAN dataset and comparing its performance with modern Spotify data.
Table of Contents

Overview
Research Question
Dataset Information
Methodology
Results
Discussion
Installation
Usage
Project Structure
Future Work
Contributing
License
Acknowledgments
References

Overview
This project investigates the practicality and validity of using the GTZAN dataset for music genre classification in today's musical landscape. With the rapid growth of digital music and the increasing importance of personalized recommendation systems, this research evaluates whether traditional datasets remain relevant for modern music classification tasks.
Key Features

CNN-based Classification: Implementation of Convolutional Neural Networks for music genre classification
Comparative Analysis: Performance comparison between GTZAN dataset and modern Spotify data
Feature Extraction: Utilization of Mel-frequency cepstral coefficients (MFCCs) for audio feature representation
Comprehensive Evaluation: Multiple metrics including accuracy, precision, recall, and F1-score

Research Question

Is the GTZAN dataset a useful and valid dataset to classify music genre using deep learning models, and can it be used to classify music genre in the current musical landscape?

Dataset Information
GTZAN Dataset (Primary Dataset)

Size: 1,000 audio files (30 seconds each)
Genres: 10 distinct genres with equal distribution
Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock


Distribution: 100 tracks per genre (balanced dataset)
Year: Originally created in 2002 by George Tzanetakis
Significance: Benchmark dataset for music information retrieval (MIR)
Limitations: Early 2000s music, potential quality/labeling issues, mislabeled samples

Best of Spotify 2022 Dataset (Contemporary Comparison)

Size: 500 audio files
Collection Method: Top 50 songs per genre via Spotify API
Purpose: Evaluate GTZAN model performance on contemporary music
Significance: Represents current popular music trends and production techniques
Distribution: Balanced across the same 10 genres as GTZAN
Limitations: Small sample size (50 songs/genre), inconsistencies in genre playlists, unavailability of some song previews

Methodology
This research employs a quantitative experimental design with four distinct stages:
1. Data Collection & Preprocessing

GTZAN Dataset: 1,000 audio tracks (30 seconds each) across 10 genres
Equal distribution: 100 tracks per genre
Genres: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock


Spotify 2022 Dataset: Top 50 songs per genre (500 total tracks)
Contemporary music representation
Collected via Spotify API
Challenges: Some songs duplicated or removed due to unavailable previews


Audio Processing: 
Standardized to WAV format with consistent sampling rate
Feature extraction using Mel-frequency cepstral coefficients (MFCCs) and Chroma features
Data normalization and preprocessing



2. Model Architecture & Optimization

Primary Model: Convolutional Neural Network (CNN)
Feature Input: 
Mel-frequency cepstral coefficients (MFCCs) - primary features
Chroma features - harmonic information
Combined features capture timbral, textural, and harmonic characteristics


CNN Architecture:
Convolutional Layers: Local pattern detection with spatial preservation
Pooling Layers: Dimensional reduction and robustness enhancement
Batch Normalization: Training stabilization and acceleration
Activation Functions: ReLU/Leaky ReLU (hidden layers), Softmax (output layer)
Optimizer: Adam optimizer with adaptive learning rate


Implementation Stack:
TensorFlow: Core machine learning framework
Keras: High-level neural network API
Librosa: Audio processing and feature extraction


Output: 10-class genre classification with probability distributions

3. Training & Validation Strategy

Three Model Variants:
GTZAN CNN: Trained on full GTZAN dataset (100 songs/genre)
Balanced GTZAN CNN: Trained on balanced dataset (50 songs/genre)
Spotify CNN: Trained on Spotify 2022 dataset (50 songs/genre)


Balanced Comparison: Equal sample sizes for fair cross-dataset evaluation
Performance Monitoring: Validation splits to prevent overfitting
Cross-Dataset Testing: Models tested across different temporal datasets

4. Evaluation Metrics

Quantitative Metrics: Accuracy, Precision, Recall, F1-score
Visual Analysis: Confusion matrices for genre-specific performance
Cross-dataset Analysis: Generalization capability assessment
Comparative Performance: Historical vs. contemporary music classification

Results
The comprehensive experimental analysis provides concrete performance metrics and reveals critical insights about dataset temporal validity:
Quantitative Performance Results
Model Performance Comparison



Model
Dataset
Accuracy
Key Observations



GTZAN CNN
GTZAN Test
74.75%
Strong baseline performance on historical data


Balanced GTZAN CNN
Balanced GTZAN
65.60%
Fair comparison baseline (50 songs/genre)


Spotify CNN
Spotify Test
70.67%
Effective on contemporary music


GTZAN CNN
Spotify Data
41.76%
Significant cross-temporal performance degradation


Genre-Specific Performance Insights

Classical Music: Consistently high accuracy across all models (e.g., GTZAN CNN on Spotify: 0.98 precision, 0.86 recall)
Blues, Country, Rock: Higher misclassification rates due to overlapping features, especially in cross-dataset testing
Contemporary Genres: Better captured by Spotify-trained models
Cross-Dataset: Pronounced performance variation between historical and modern music
Spotify CNN Challenges: Struggled with blues and rock, similar to GTZAN models, indicating shared genre overlap issues
GTZAN CNN on Spotify: Notable misclassifications (e.g., blues/country, hip-hop/reggae) and high false-positive rates

Critical Research Findings
Dataset Temporal Validity

20+ Year Gap Impact: GTZAN (2002) vs Spotify (2022) shows significant performance degradation (0.7475 to 0.4176 for GTZAN CNN)
Balanced Dataset Necessity: 50 songs/genre provides fairer cross-dataset comparison
Genre Evolution: Modern production techniques not captured in historical training data
Spotify Dataset Limitations: Small sample size, playlist inconsistencies, and preview unavailability impacted performance

Model Architecture Effectiveness

CNN Architecture: Proven effective for audio classification tasks
MFCC + Chroma Features: Successfully capture genre-distinguishing characteristics
Transfer Learning Limitations: Historical models poorly generalize to contemporary music

Methodological Validation

Cross-Dataset Evaluation: Essential for assessing real-world applicability
Balanced Sampling: Critical for fair performance comparisons
Multi-Metric Analysis: Accuracy, precision, recall, F1-score provide comprehensive assessment

Comparative Analysis

GTZAN CNN vs. Balanced GTZAN CNN: The GTZAN CNN (0.7475 accuracy) outperformed the Balanced GTZAN CNN (0.6560 accuracy), likely due to overfitting to the larger, original GTZAN dataset. The balanced model provides a fairer comparison with Spotify CNN.
GTZAN CNN vs. Spotify CNN: GTZAN CNN (0.7475 on GTZAN) vs. Spotify CNN (0.7067 on Spotify) shows comparable performance on respective datasets, but GTZAN CNN's accuracy drops to 0.4176 on Spotify data, indicating poor generalization to modern music.
Spotify CNN Performance: Better suited for contemporary music, capturing modern production nuances, though it struggles with genres like blues and rock due to feature overlap.

Statistical Analysis
The following chart illustrates the accuracy scores across different models and datasets:
{
  "type": "bar",
  "data": {
    "labels": ["GTZAN CNN on GTZAN", "GTZAN CNN on Balanced GTZAN", "Spotify CNN on Spotify", "GTZAN CNN on Spotify"],
    "datasets": [{
      "label": "Accuracy",
      "data": [0.747499, 0.656000, 0.706667, 0.417556],
      "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
      "borderColor": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
      "borderWidth": 1
    }]
  },
  "options": {
    "scales": {
      "y": {
        "beginAtZero": true,
        "title": {
          "display": true,
          "text": "Accuracy"
        }
      },
      "x": {
        "title": {
          "display": true,
          "text": "Model and Dataset"
        }
      }
    },
    "plugins": {
      "title": {
        "display": true,
        "text": "Model Accuracy Comparison Across Datasets"
      }
    }
  }
}


Key Insights:
The drop from 0.747499 to 0.656000 for GTZAN CNN on balanced data suggests sensitivity to dataset composition and potential overfitting.
The significant drop to 0.417556 on Spotify data highlights GTZAN CNN's limitations in generalizing to contemporary music.
Spotify CNN's 0.706667 accuracy indicates better adaptation to modern music trends.



Limitations

Small Sample Size: Spotify dataset (50 songs/genre) limits diversity and generalization.
Balanced GTZAN Dataset: Reducing to 50 songs/genre may have decreased feature diversity.
Song Preview Unavailability: Duplication or removal of songs in Spotify dataset affected representativeness.
Playlist Inconsistencies: Lack of "Best of 2022" playlists for some genres led to less representative data.
Single Model Focus: Only CNNs were tested due to time constraints; other models (e.g., RNNs, transformers) could offer insights.
Single Accuracy Score: Limited to one run per model, lacking robustness from multiple train-test splits or hyperparameter tuning.

Results Summary

GTZAN CNN: Strong performance on GTZAN dataset (0.7475 accuracy), excelling in classical music but struggling with blues, country, and rock.
Balanced GTZAN CNN: Fairer comparison with 0.6560 accuracy, reflecting reduced overfitting but lower performance.
Spotify CNN: Effective on Spotify dataset (0.7067 accuracy), with similar strengths (classical) and weaknesses (blues, rock).
Cross-Dataset Performance: GTZAN CNN's 0.4176 accuracy on Spotify data underscores its limitations for modern music.
Research Question Insight: The GTZAN dataset is less effective for contemporary music classification due to temporal gaps and evolving genre characteristics.

Discussion
Discussion Overview
This section critically examines the findings to address the research question: Is the GTZAN dataset a useful and valid dataset to classify music genre using machine learning models, and can it be used to classify music genre in the current musical landscape? The discussion is structured into sub-sections to provide a systematic analysis:

Performance of the GTZAN CNN Model: Analyzes strengths and weaknesses.
Weaknesses of the GTZAN Dataset: Identifies factors limiting its applicability.
Recommendations: Suggests improvements for the dataset and classification tasks.
Further Research: Explores future research directions.

Performance of the GTZAN CNN Model
Strengths

Overall Accuracy: The GTZAN CNN model achieved a 0.7475 accuracy on the GTZAN dataset, demonstrating its potential for effective genre classification. This suggests that the dataset contains informative samples for certain genres.
High Performance in Specific Genres: The model excelled in classifying classical music, with high precision (0.98) and recall (0.86) even on the Spotify dataset, indicating that the GTZAN dataset captures distinct features for some genres.
Robustness to Noise and Variations: Despite variations in audio quality from diverse sources (CDs, radio, internet), the model maintained reasonable performance, highlighting its ability to handle noisy data.

Weaknesses

Inconsistent Performance Across Genres: The model struggled with genres like blues, country, and rock, showing high misclassification rates due to overlapping features, indicating incomplete representation in the GTZAN dataset.
Difficulty Generalizing to Newer Datasets: The significant accuracy drop to 0.4176 on the Spotify dataset suggests that the GTZAN dataset does not adequately prepare models for contemporary music classification.
Sensitivity to Mislabelled Samples: Reported mislabeling in the GTZAN dataset likely contributed to inconsistent genre performance and poor generalization.
Limitations in Capturing Subjectivity: The inherent subjectivity of genre classification, with overlapping characteristics, posed challenges for the model, particularly for genres like blues and country or hip-hop and reggae.

Weaknesses of the GTZAN Dataset
The GTZAN dataset's limitations include its age (created in 2002), leading to a lack of representation of modern music trends and sub-genres. Mislabeling issues hinder the model's ability to learn accurate genre features. The dataset's limited diversity fails to capture the full spectrum of genre characteristics, particularly for genres with overlapping features. Additionally, its static nature does not account for genre evolution, reducing its applicability to contemporary music classification tasks.
Implications of the GTZAN CNN Model's Performance for the GTZAN Dataset
The GTZAN CNN model's performance provides insights into the dataset's utility and limitations. Its strengths in specific genres and robustness to noise suggest that the dataset is valuable for certain classification tasks. However, its inconsistent performance across genres, poor generalization to modern datasets, sensitivity to mislabeling, and challenges with subjective genre boundaries reveal significant shortcomings. These findings indicate that the GTZAN dataset, while historically significant, may not fully align with the contemporary music landscape, necessitating updates and enhancements to improve its applicability.
Recommendations
To address the identified limitations, the following recommendations are proposed:

Updating the GTZAN Dataset: Incorporate recent songs and diverse sub-genres to reflect current music trends, with regular updates to maintain relevance.
Combining Multiple Datasets: Merge GTZAN with datasets like FMA or MSD to create a more comprehensive and representative training set.
Implementing Data Augmentation: Use techniques like time-stretching or pitch-shifting to increase dataset size and diversity, improving model robustness.
Collaborating with Experts: Work with musicologists and industry professionals to capture genre subjectivity and enhance classification accuracy.
Developing a Yearly Updated Dataset: Create or update a dataset annually with popular songs, curated with expert input, to ensure relevance.
Utilizing Streaming Services: Leverage metadata and playlists from platforms like Spotify to build comprehensive datasets reflecting current trends.

Further Research
Future research should address the following areas:

Alternative Datasets: Investigate datasets like FMA, MSD, or Spotify API data to compare their effectiveness with GTZAN.
Dynamic Datasets: Develop datasets that evolve with new music trends to maintain classification relevance.
Additional Features: Incorporate mood, tempo, energy, and lyrical content to enhance classification performance.
Alternative Models: Explore RNNs, transformers, or transfer learning to improve handling of genre subjectivity.
Cross-Domain Classification: Develop models for hybrid genres to reflect modern music complexity.
Personalization: Integrate user preferences and listening history for tailored classification systems.

Installation
Prerequisites

Python 3.8+
TensorFlow 2.x
Keras (included with TensorFlow)
librosa (audio processing)
numpy
pandas
matplotlib
scikit-learn
Spotify API credentials (for dataset collection)

Setup
# Clone the repository
git clone https://github.com/yourusername/gtzan-music-classification.git
cd gtzan-music-classification

# Install dependencies
pip install tensorflow librosa numpy pandas matplotlib scikit-learn spotipy

# Download GTZAN dataset (follow dataset provider's instructions)
# Configure Spotify API credentials in config.py

Usage
Basic Classification
from src.models import GTZANCNNModel
from src.preprocessing import extract_features

# Load and preprocess data
features = extract_features('path/to/audio/file.wav')

# Load trained model
model = GTZANCNNModel.load_model('models/gtzan_cnn.h5')

# Predict genre
prediction = model.predict(features)
print(f"Predicted genre: {prediction}")

Audio Feature Extraction
import librosa
import numpy as np
from src.features import extract_audio_features

# Extract MFCC and Chroma features
def extract_features(audio_path, n_mfcc=13, n_chroma=12):
    """
    Extract MFCC and Chroma features from audio file
    """
    y, sr = librosa.load(audio_path, duration=30)  # 30-second clips
    
    # MFCC features (primary)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Chroma features (harmonic information)  
    chroma = librosa.feature.chroma(y=y, sr=sr, n_chroma=n_chroma)
    
    # Combine features
    features = np.concatenate([mfccs, chroma], axis=0)
    return np.mean(features.T, axis=0)

# Process dataset
features = extract_features('path/to/audio/file.wav')

Model Training & Evaluation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape, num_classes=10):
    """
    Create CNN model for music genre classification
    """
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(64, 3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, 3, activation='relu'),
        BatchNormalization(),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # 10 genre classes
    ])
    
    # Adam optimizer with adaptive learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Train models
gtzan_model = create_cnn_model(input_shape=(25,))  # MFCC + Chroma features
gtzan_model.fit(X_train_gtzan, y_train_gtzan, 
                validation_split=0.2, 
                epochs=100, 
                batch_size=32)

Cross-Dataset Performance Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_cross_dataset(model, X_test, y_test, dataset_name):
    """
    Comprehensive evaluation with multiple metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    
    # Classification report
    report = classification_report(y_true_classes, y_pred_classes, 
                                 target_names=genre_labels)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    print(f"\n{dataset_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Classification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_labels, yticklabels=genre_labels)
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.show()
    
    return accuracy, report, cm

# Example: Evaluate GTZAN model on different datasets
gtzan_accuracy, _, _ = evaluate_cross_dataset(
    gtzan_model, X_test_gtzan, y_test_gtzan, "GTZAN Model on GTZAN Data"
)

spotify_cross_accuracy, _, _ = evaluate_cross_dataset(
    gtzan_model, X_test_spotify, y_test_spotify, "GTZAN Model on Spotify Data"
)

print(f"\nPerformance Drop: {gtzan_accuracy - spotify_cross_accuracy:.4f}")

Project Structure
gtzan-music-classification/
├── data/
│   ├── gtzan/
│   ├── spotify/
│   └── processed/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py
│   │   └── base_model.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── feature_extraction.py
│   │   └── data_augmentation.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   ├── train.py
│   └── predict.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── models/
│   ├── gtzan_cnn.h5
│   └── spotify_cnn.h5
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
├── README.md
└── LICENSE

Future Work
Based on the performance results and identified limitations, the following research directions are recommended:

Dataset Enhancement and Modernization: Develop comprehensive modern music datasets post-2020, spanning multiple decades to study genre evolution, with balanced representation and standardized audio quality.
Advanced Feature Engineering: Integrate multi-modal features (audio, lyrics, album art), temporal features (tempo, rhythm), emotional/mood features, and production-based features (sampling, auto-tuning).
Model Architecture Improvements: Implement transfer learning, attention mechanisms, ensemble methods, and multi-label classification for hybrid genres.
Cross-Temporal Adaptation: Use domain adaptation, continuous learning, temporal weighting, and genre evolution modeling to bridge historical and contemporary gaps.
Evaluation Methodology: Conduct real-world testing on streaming data, incorporate musicologist validations, address genre subjectivity, and expand to non-Western genres.
Practical Applications: Develop models for real-time music recommendation, playlist generation, music discovery, and educational tools.

Contributing
Contributions are welcome. Please submit a Pull Request. For major changes, open an issue first to discuss proposed changes.
Development Guidelines

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Author: Jupiter Mellen (SID: 10815962)
Supervisor: Beate Grawemeyer
Institution: Faculty of Engineering, Environment and Computing, Coventry University
Course: 6001CEM Individual Project
Ethics Application: P146117

Special Thanks

Tzanetakis and Cook for the original GTZAN dataset
Spotify for providing API access to modern music data
The open-source community for various tools and libraries used in this project

References

Ajoodha, R., & Dhevan S, L. (2022). Music Genre Classification: A Comparative Study Between Deep Learning and Traditional Machine Learning Approaches. In X. S. Yang, Proceedings of Sixth International Congress on Information and Communication Technology (pp. 239-247). Singapore: Springer Singapore.
Elbir, A., & Aydin, N. (2020). Music genre classification and music recommendation by using Deep Learning. Electronics Letters, 627-629.
Fabbri, F. (2012). Genre theories and their applications in the historical and analytical study of popular music: a commentary on my publications. Huddersfield: University Huddersfield Repository.
Ghildiyal, A., Singh, K., & Sharma, S. (2020). Music genre classification using machine learning. 2020 4th International Conference on Electronics, Communication and Aerospace Technology (ICECA), 1368-1372.
Kamalesh, P., Dipika, S., & Angela, Y. (2020). Rethinking CNN Models for Audio Classification. Singapore: arXiv.
Ndou, N., Ajoodha, R., & Jadhav, A. (2021). Music genre classification: A review of deep-learning and traditional machine-learning approaches. 2021 IEEE International IOT, Electronics and Mechatronics Conference (IEMTRONICS), 1-6.
Pablo, M.-H. (2017). A Theory of the Musical Genre: The Three-Phase Cycle. Proceedings of the 10th International Conference of Students of Systematic Musicology (pp. 13-15). London: Peter M. C. Harrison (Ed.).
Pelchat, N., & Gelowitz, C. (2020). Neural Network Music Genre Classification. Canadian Journal of Electrical and Computer Engineering, 170-173.
Sturm, B. L. (2012). An Analysis of the GTZAN Music Genre Dataset. Proceedings of the second international ACM workshop on Music information retrieval with user-centered and multimodal (pp. 7–12). Nara, Japan: Association for Computing Machinery.
Tagg, P. (1982). Analysing popular music: theory, method and practice. Popular Music, 37-67.


Note: This research highlights the importance of using contemporary datasets for modern music classification tasks while acknowledging the historical significance of the GTZAN dataset in the field of music information retrieval. The findings underscore the need for updated datasets and advanced methodologies to address the evolving nature of music genres.
