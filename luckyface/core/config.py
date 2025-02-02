from ..vlms.prompts import ANALYSIS_PROMPT, FALLBACK_PROMPT
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseSettings):
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    camera_id: int = 0
    analysis_prompt: str = ANALYSIS_PROMPT
    fallback_prompt: str = FALLBACK_PROMPT
    min_face_size: tuple = (30, 30)
    enhancement_enabled: bool = True
    debug_mode: bool = False
    
    class Config:
        env_file = ".env"