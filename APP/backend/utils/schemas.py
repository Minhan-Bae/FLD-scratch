from typing import List, Optional

from fastapi import Body, File, Form, UploadFile
from pydantic import BaseModel

class Message(BaseModel):
    message: str


class FileName(BaseModel):
    file_name: str


class Coordinate(BaseModel):
    number: Optional[int] = Body(None, description="Point number", ge=0)
    name: Optional[str] = Body(None, description="Point name")
    y: int = Body(..., description="Vertical position of the point on the image", ge=0)
    x: int = Body(..., description="Horizontal position of the point on the image", ge=0)


class Landmark(BaseModel):
    coordinates: List[Coordinate]


class ImagesForm:
    def __init__(
        self,
        image: UploadFile = File(..., description="input image"),
    ) -> None:
        self.image = image