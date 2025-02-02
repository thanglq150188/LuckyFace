from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class AnalysisResult(BaseModel):
    timestamp: datetime
    face_detected: bool
    analysis: Optional[Dict[str, Any]]
    error_message: Optional[str]
    tokens_used: Optional[int]

    class Config:
        arbitrary_types_allowed = True