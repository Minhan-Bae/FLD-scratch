from typing import Dict
from fastapi import Depends, FastAPI

import responses
import utils.schemas as schemas
from models.xception import xception_Net_54
from utils.inference import inference_landmark_detection

model = xception_Net_54()

app = FastAPI()

@app.get(
    "/",
    tags=["home"],
    response_model=schemas.Message,
    summary="Display the homepage",
)
def go_home() -> Dict:
    return {"message": "homepage"}


@app.post("/prediction", tags=["main"])
async def get_prediction(images: schemas.ImagesForm = Depends(),) -> Dict:

    crop_image = images.image
    landmarks = inference_landmark_detection(model, crop_image)

    return landmarks


def main():
    import uvicorn
    uvicorn.run("back:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()