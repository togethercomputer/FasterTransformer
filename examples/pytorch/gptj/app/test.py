import requests

prompt = "Where is Zurich?"
top_p = 1.0
temperature = 0.5
max_new_tokens = 32


my_post_dict = {
    "model": "Together-gpt-JT-6B-v1",
    "prompt": prompt,
    "top_p": float(top_p),
    "temperature": float(temperature),
    "max_tokens": int(max_new_tokens),
    "stop": ['\n']
}

print(my_post_dict)
response = requests.get("https://staging.together.xyz/api/inference", params=my_post_dict).json()