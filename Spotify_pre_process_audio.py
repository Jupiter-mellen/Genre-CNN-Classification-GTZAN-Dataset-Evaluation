import json, os, math
import librosa

DATASET_PATH = "Spotify/Genres"
JSON_PATH = "data_10_spotify.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def create_mfcc(DATASET_PATH, JSON_PATH, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    data_dict = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)):
        if dirpath != DATASET_PATH:
            semantic_label = dirpath.split("\\")[-1]
            data_dict["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label.upper()} ")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sample_rate, = librosa.load(file_path, sr=SAMPLE_RATE)

                for d in range(num_segments):
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # Pass 'y' and 'sr' as keyword arguments
                    mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=SAMPLE_RATE, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data_dict["mfcc"].append(mfcc.tolist())
                        data_dict["labels"].append(i-1)
                        print(f"{file_path}, segment: {d+1}")

    with open(JSON_PATH, "w") as fp:
        json.dump(data_dict, fp, indent=4)

create_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
