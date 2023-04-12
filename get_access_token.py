import base64
import requests

# Set up authentication headers
client_id = "03392ce9389a4e0d8121d08019e21986"
client_secret = "37a827682f22495caf90d8e3e721f710"
auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")
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
response = requests.post(url, headers=headers, data=params)
response_json = response.json()

# Extract access token from response
access_token = response_json["access_token"]
print(access_token)