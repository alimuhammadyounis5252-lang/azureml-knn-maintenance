import urllib.request
import json
import os

# Load request data
with open('deployment/sample-request.json', 'r') as f:
    request_data = json.load(f)

body = str.encode(json.dumps(request_data))

# Set endpoint URL and Key (replace with your Azure ML endpoint details or environment variables)
url = os.environ.get('ENDPOINT_URL', 'YOUR_ENDPOINT_URL_HERE')
api_key = os.environ.get('ENDPOINT_KEY', 'YOUR_ENDPOINT_KEY_HERE')

headers = {'Content-Type': 'application/json'}
if api_key:
    headers['Authorization'] = f'Bearer {api_key}'

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    print("Response:", result.decode('utf-8'))
except urllib.error.HTTPError as error:
    print(f"The request failed with status code: {error.code}")
    print(error.read().decode("utf-8", "ignore"))
