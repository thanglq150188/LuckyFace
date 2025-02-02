from openai import OpenAI
from PIL import Image
import io
from typing import Dict, Any
from ..core.config import Settings

class FaceAnalyzer:
    def __init__(self, settings: Settings):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.settings = settings

    def analyze_face(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze face using OpenAI's vision model."""
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create base64 string
        import base64
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": self.settings.analysis_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            max_tokens=500
        )
        
        return {
            'analysis': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens
        }