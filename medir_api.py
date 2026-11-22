import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from api import app

JPEG_HEADERS = {'Content-Type': 'image/jpeg'}


def measure(client, images, workers):
    def request(image):
        start = time.perf_counter()
        response = client.post('/predict', content=image, headers=JPEG_HEADERS)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        if not math.isfinite(response.json()['edad_estimada']):
            raise ValueError('La API devolvió una edad no finita.')
        return elapsed_ms

    with ThreadPoolExecutor(max_workers=workers) as executor:
        start = time.perf_counter()
        timings = list(executor.map(request, images))
        elapsed = time.perf_counter() - start
    sorted_timings = sorted(timings)
    return {
        'requests': len(timings),
        'concurrency': workers,
        'p50_ms': round(statistics.median(timings), 3),
        'p95_ms': round(sorted_timings[math.ceil(len(timings) * 0.95) - 1], 3),
        'requests_per_second': round(len(timings) / elapsed, 2),
    }


def main():
    parser = argparse.ArgumentParser(description='Comprueba y mide FastAPI con imágenes del conjunto de prueba.')
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('rendimiento_api.json'))
    args = parser.parse_args()

    with ZipFile(args.archive) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith('20-50/test/') and name.endswith('.jpg'))
        if len(names) < 120:
            raise ValueError('El conjunto de prueba tiene menos de 120 entradas.')
        images = [archive.read(names[i * len(names) // 120]) for i in range(120)]

    with TestClient(app) as client:
        assert client.get('/health').json() == {'status': 'ready'}
        assert client.post('/predict', content=images[0]).status_code == 415
        assert client.post('/predict', content=b'no es un JPEG', headers=JPEG_HEADERS).status_code == 422
        assert client.post('/predict', content=b'0' * (5 * 1024 * 1024 + 1), headers=JPEG_HEADERS).status_code == 413
        for image in images[:8]:
            client.post('/predict', content=image, headers=JPEG_HEADERS).raise_for_status()
        measurements = [measure(client, images, workers) for workers in (1, 4)]

    results = {'transport': 'FastAPI TestClient en proceso', 'measurements': measurements}
    report = json.dumps(results, ensure_ascii=False, indent=2)
    args.output.write_text(report + '\n', encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
