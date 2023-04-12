# Import necessary libraries
import os
import requests
import base64

# Set up authentication headers
client_id = "03392ce9389a4e0d8121d08019e21986"
client_secret = "37a827682f22495caf90d8e3e721f710"

# Encode client ID and secret using base64
auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")

# Set authorization header
headers = {
    "Authorization": f"Basic {auth_header}"
}

# Define request parameters
grant_type = "client_credentials"

# Make request to obtain access token
url = "https://accounts.spotify.com/api/token"
params = {
    "grant_type": grant_type
}

# Send POST request to the Spotify API with authorization headers and request parameters
response = requests.post(url, headers=headers, data=params)

# Extract access token from response
response_json = response.json()
access_token = response_json["access_token"]

# Set up headers for Spotify API
headers = {
    "Authorization": f"Bearer {access_token}"
}

# Define the playlists for each genre as a dictionary, where the key is the genre name and the value is the Spotify playlist ID
genre_playlists = {
    "blues": "37i9dQZF1DWYi488IywmOA",
    "classical": "62n7TtrAWY1BeNg54yigFe",
    "country": "37i9dQZF1DWYyZ38lseF2K",
    "disco": "37i9dQZF1DX1MUPbVKMgJE",
    "hiphop": "37i9dQZF1DWWEncNAQJJkE",
    "jazz": "37i9dQZF1DX5LYxFep0J7E",
    "metal": "37i9dQZF1DX5FZ0gGkvIRf",
    "pop": "37i9dQZF1DX5dpn9ROb26T",
    "reggae": "37i9dQZF1DX2oc5aN4UDfD",
    "rock": "37i9dQZF1DX6KANutsQaVe",
}

# Set up Spotify API URL
spotify_api = "https://api.spotify.com/v1"

# Define function to download preview MP3 files from Spotify
def download_preview(url, folder, filename):
    try:
        # Send GET request to download MP3 file
        response = requests.get(url)
        # Save MP3 file to specified folder with specified filename
        with open(os.path.join(folder, filename), "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        # If there was an error downloading the MP3 file, print error message and return False
        print(f"Error downloading {filename}: {e}")
        return False

# Loop through each genre and playlist, creating a folder for each genre and downloading MP3 files of song previews to the corresponding folder
for genre, playlist_id in genre_playlists.items():
    # Define URL for playlist tracks
    playlist_url = f"{spotify_api}/playlists/{playlist_id}/tracks"
    # Send GET request to Spotify API to get playlist tracks
    response = requests.get(playlist_url, headers=headers)
    # Convert response to JSON
    response_json = response.json()
    # Create genre folder if it doesn't exist
    folder = os.path.join("Spotify", "Genres", genre)
    os.makedirs(folder, exist_ok=True)

    # Initialize song counter
    song_counter = 1
    for item in response_json["items"]:
        # Get track information
        track = item["track"]
        # Get preview URL for track
        preview_url = track["preview_url"]
        # Define filename for downloaded MP3 file
        filename = f"{genre}_{song_counter}.mp3"

        if preview_url:
            # Download MP3 file
            print(f"Downloading {genre}: {filename}")
            success = download_preview(preview_url, folder, filename)
            if success:
                song_counter += 1
        else:
            # If there is no preview available, print error message
            print(f"No preview available for {genre}: {filename}")
