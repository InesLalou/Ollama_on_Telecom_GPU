import torch
from ollama import Client

# test ollama
client = Client(host='http://localhost:11434')

response = client.chat(
    model='mistral',
    messages=[
        {"role": "user", "content": "Explique-moi la théorie de la relativité en termes simples."}
    ]
)

print(response['message']['content'])
