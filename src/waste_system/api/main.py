from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from waste_system import __version__
from waste_system.config import get_settings, project_path
from waste_system.inference.classifier import ModelUnavailableError, TorchscriptClassifier
from waste_system.inference.decision_engine import DecisionEngine
from waste_system.inference.detector import DetectorUnavailableError, UltralyticsWasteDetector
from waste_system.inference.multi_object import MultiObjectPipeline
from waste_system.inference.preprocessing import assess_quality, decode_image, prepare_rgb
from waste_system.llm.advisor import TemplateAdvisor
from waste_system.schemas import DecisionRequest, MultiPredictionResponse, PredictionResponse


app = FastAPI(
    title="智慧垃圾辨識系統 API",
    version=__version__,
    description="OpenCV、CNN、感測器決策融合與回收建議 API",
)


@lru_cache(maxsize=1)
def classifier() -> TorchscriptClassifier:
    settings = get_settings()
    return TorchscriptClassifier(
        model_path=project_path(settings["model"]["path"]),
        classes=settings["classes"],
        device=settings["model"]["device"],
    )


@lru_cache(maxsize=1)
def detector() -> UltralyticsWasteDetector:
    settings = get_settings()["detector"]
    return UltralyticsWasteDetector(
        model_path=project_path(settings["path"]),
        device=settings["device"],
        confidence=settings["confidence"],
        iou=settings["iou"],
        maximum_objects=settings["maximum_objects"],
    )


def run_decision(request: DecisionRequest):
    settings = get_settings()
    engine = DecisionEngine(settings["decision"])
    return engine.decide(request, minimum_image_quality=settings["image"]["minimum_quality"])


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "version": __version__,
        "classifier_available": classifier().available,
        "detector_available": detector().available,
        "multi_object_ready": classifier().available and detector().available,
    }


@app.get("/scanner", include_in_schema=False)
def scanner() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "scanner.html")


@app.get("/local-ca.crt", include_in_schema=False)
def local_ca_certificate() -> FileResponse:
    certificate_path = project_path("certs/trash-identify-ca.crt")
    if not certificate_path.is_file():
        raise HTTPException(status_code=404, detail="Local CA certificate has not been generated")
    return FileResponse(
        certificate_path,
        media_type="application/x-x509-ca-cert",
        filename="trash-identify-ca.crt",
    )


@app.post("/v1/decide", response_model=PredictionResponse)
def decide(request: DecisionRequest) -> PredictionResponse:
    result = run_decision(request)
    return PredictionResponse(
        result=result,
        probabilities=request.probabilities,
        image_quality=request.image_quality,
        advice=TemplateAdvisor().advise(result, request.region),
        region=request.region,
    )


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    region: str = Form(os.getenv("WASTE_REGION", "Taiwan")),
    metal_detected: bool | None = Form(None),
    magnetic: bool | None = Form(None),
    weight_grams: float | None = Form(None),
) -> PredictionResponse:
    settings = get_settings()
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")
    try:
        image = decode_image(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_settings = settings["image"]
    quality = assess_quality(
        image,
        blur_threshold=image_settings["blur_laplacian_threshold"],
        minimum_brightness=image_settings["minimum_brightness"],
        maximum_brightness=image_settings["maximum_brightness"],
    )
    rgb = prepare_rgb(image, image_settings["width"], image_settings["height"])
    try:
        probabilities = classifier().predict(rgb)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    request = DecisionRequest(
        probabilities=probabilities,
        image_quality=quality.score,
        multi_view_agreement=1.0,
        metal_detected=metal_detected,
        magnetic=magnetic,
        weight_grams=weight_grams,
        region=region,
    )
    result = run_decision(request)
    return PredictionResponse(
        result=result,
        probabilities=probabilities,
        image_quality=quality.score,
        advice=TemplateAdvisor().advise(result, region),
        region=region,
    )


@app.post("/v1/predict-multiple", response_model=MultiPredictionResponse)
async def predict_multiple(
    file: UploadFile = File(...),
    region: str = Form(os.getenv("WASTE_REGION", "Taiwan")),
) -> MultiPredictionResponse:
    settings = get_settings()
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")
    try:
        image = decode_image(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image_settings = settings["image"]
    quality = assess_quality(
        image,
        blur_threshold=image_settings["blur_laplacian_threshold"],
        minimum_brightness=image_settings["minimum_brightness"],
        maximum_brightness=image_settings["maximum_brightness"],
    )
    height, width = image.shape[:2]
    if quality.score < image_settings["minimum_quality"]:
        return MultiPredictionResponse(
            image_width=width,
            image_height=height,
            image_quality=quality.score,
            objects=[],
            region=region,
        )

    pipeline = MultiObjectPipeline(detector(), classifier(), settings)
    try:
        return pipeline.predict(image, image_quality=quality.score, region=region)
    except (DetectorUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
