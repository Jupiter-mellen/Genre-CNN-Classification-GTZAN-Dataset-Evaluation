import os
import random
import shutil

# Define the path to the genres folder
genres_folder = "Spotify/Genres"

# Function to randomly duplicate a song in a genre folder
def duplicate_random_song(genre_folder):
    files = os.listdir(genre_folder)
    random_song = random.choice(files)
    src_file = os.path.join(genre_folder, random_song)
    new_song_number = len(files) + 1
    dst_file = os.path.join(genre_folder, f"{genre}_{new_song_number}.mp3")
    shutil.copy(src_file, dst_file)

# Function to remove extra songs above genre_50 in a genre folder
def remove_extra_songs(genre_folder):
    files = os.listdir(genre_folder)
    files_to_remove = [f for f in files if int(f.split("_")[1].split(".")[0]) > 50]
    for file_to_remove in files_to_remove:
        os.remove(os.path.join(genre_folder, file_to_remove))

# Iterate through each genre folder
for genre in os.listdir(genres_folder):
    genre_folder = os.path.join(genres_folder, genre)
    
    # Check if the folder is a directory (to ignore any non-folder items)
    if os.path.isdir(genre_folder):
        song_count = len(os.listdir(genre_folder))

        # If there are less than 50 songs, duplicate random songs until there are 50
        if song_count < 50:
            while song_count < 50:
                duplicate_random_song(genre_folder)
                song_count += 1
        # If there are more than 50 songs, remove all songs with names above genre_50
        elif song_count > 50:
            remove_extra_songs(genre_folder)
