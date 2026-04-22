import base64
import json
import sys
from openai import OpenAI

client = OpenAI()

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ---- Get image path from command line ----
if len(sys.argv) < 2:
    print("Usage: python image_credit_analysis.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# ---- Encode image ----
base64_image = encode_image(image_path)

# ---- Prompt ----
prompt = """
You are an AI assistant helping in field-based credit assessment using images.

Return STRICT JSON with:
- image_description
- image_type
- lifestyle_indicators
- income_estimation
- credit_assessment_signals
- overall_summary

Return ONLY JSON.
"""

# ---- API call ----
response = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ]    
)

output_text = response.choices[0].message.content

# ---- Print JSON ----
try:
    parsed = json.loads(output_text)
    print(json.dumps(parsed, indent=2))
except:
    print(output_text)