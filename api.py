import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

IMAGE_SIZE = (64, 64)
MAX_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 12_000_000
IMAGE_FORMATS = {'image/jpeg': 'JPEG', 'image/png': 'PNG'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    import tensorflow as tf

    model_path = Path(os.environ.get('MODEL_PATH', Path(__file__).with_name('modelo_edad.keras')))
    model = tf.keras.models.load_model(model_path, compile=False)

    @tf.function(input_signature=[tf.TensorSpec((None, *IMAGE_SIZE, 3), tf.uint8)])
    def infer(batch):
        return model(batch, training=False)

    infer(tf.zeros((1, *IMAGE_SIZE, 3), dtype=tf.uint8))
    app.state.infer = infer
    yield


app = FastAPI(title='Predicción de edad', version='1.0.0', lifespan=lifespan)


@app.get('/health')
def health():
    return {'status': 'ready'}


def predict_age(content: bytes, expected_format: str) -> float:
    try:
        with Image.open(io.BytesIO(content), formats=[expected_format]) as image:
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, 'La imagen excede el límite de píxeles.')
            resized = image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            batch = np.asarray(resized, dtype=np.uint8)[None, ...]
    except Image.DecompressionBombError as error:
        raise HTTPException(413, 'La imagen excede el límite de píxeles.') from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(422, 'No se pudo leer una imagen del formato indicado.') from error

    return float(app.state.infer(batch).numpy()[0, 0])


@app.post('/predict')
async def predict(request: Request):
    media_type = request.headers.get('content-type', '').split(';', 1)[0].strip().lower()
    if media_type not in IMAGE_FORMATS:
        raise HTTPException(415, 'Envía el archivo como cuerpo binario con Content-Type image/jpeg o image/png.')

    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, 'La imagen excede los 5 MiB.')
        content.extend(chunk)

    age = await run_in_threadpool(predict_age, bytes(content), IMAGE_FORMATS[media_type])
    return {'edad_estimada': age}
