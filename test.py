import requests

url = 'http://localhost:9696/predict'
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'

# Sample Data
sample_data = {
"cap_diameter": 15.26,
"stem_height": 16.95,
"stem_width": 17.09,
"gill_color": "w",
"does_bruise_or_bleed": "f",
"stem_surface": "y",
"cap_shape": "x",
"habitat": "d",
"gill_attachment": "e",
"season": "w",
"ring_type": "g",
"cap_surface": "g",
"cap_color": "o",
"has_ring": "t",
"gill_spacing": "unknown",
"stem_color": "w"
}

response = requests.post(url, json=sample_data)

prediction = response.json()

print(prediction)