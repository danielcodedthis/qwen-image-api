from typing import Optional

from pydantic import BaseModel


class ImageRequest(BaseModel):
    prompt: str
    # None = use default_model from config.yaml.
    model: Optional[str] = None
    # None = use generation.negative_prompt from config.yaml.
    negative_prompt: Optional[str] = None
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
