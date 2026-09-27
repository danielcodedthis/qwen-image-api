from pydantic import BaseModel


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    response_format: str = "b64_json"
