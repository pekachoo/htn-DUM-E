from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
prompt_text = """You are a computer vision system for robotic sorting. Analyze the image and identify all objects within the bounding boxes. For each object, provide:
- Object type (e.g., apple, bottle, book, tool, etc.)
- Physical characteristics (color, size, shape, material if possible)
- Condition (damage, wear, markings, etc.)
- Sorting decision: Assign each object to the minimum number of bins needed, based on their properties. You may create as many bins as necessary, naming them Bin 1, Bin 2, ..., Bin N. Group similar objects together in the same bin if appropriate.
- Brief reasoning for the bin assignment.

Respond in a clean, easy-to-read english format. say the number you chose for n as well. 
"""

completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://i.sstatic.net/KiN7k.png"
                    },
                },
            ],
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
)

print(completion.choices[0].message.content)
