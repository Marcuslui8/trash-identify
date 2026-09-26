from __future__ import annotations

from typing import Any

from waste_system.schemas import DecisionRequest, DecisionResult, DecisionStatus, WasteClass


class DecisionEngine:
    def __init__(self, settings: dict[str, Any]) -> None:
        self.acceptance_confidence = float(settings["acceptance_confidence"])
        self.review_confidence = float(settings["review_confidence"])
        self.minimum_margin = float(settings["minimum_margin"])
        self.minimum_multi_view_agreement = float(settings["minimum_multi_view_agreement"])

    def decide(self, request: DecisionRequest, minimum_image_quality: float) -> DecisionResult:
        ranked = sorted(request.probabilities.items(), key=lambda item: item[1], reverse=True)
        predicted_class, confidence = ranked[0]
        margin = confidence - ranked[1][1]
        reasons: list[str] = []

        if request.image_quality < minimum_image_quality:
            return DecisionResult(
                final_class=WasteClass.UNKNOWN,
                confidence=confidence,
                status=DecisionStatus.RESCAN,
                reason_codes=["image_quality_too_low"],
            )

        sensor_conflict = (
            request.metal_detected is True and predicted_class not in {WasteClass.METAL, WasteClass.UNKNOWN}
        ) or (request.metal_detected is False and predicted_class == WasteClass.METAL)
        if sensor_conflict:
            reasons.append("metal_sensor_conflict")
        if request.multi_view_agreement < self.minimum_multi_view_agreement:
            reasons.append("multi_view_disagreement")
        if margin < self.minimum_margin:
            reasons.append("class_margin_too_small")

        if predicted_class == WasteClass.UNKNOWN:
            reasons.append("model_predicted_unknown")
            status = DecisionStatus.UNKNOWN
            final_class = WasteClass.UNKNOWN
        elif confidence < self.review_confidence:
            reasons.append("confidence_too_low")
            status = DecisionStatus.UNKNOWN
            final_class = WasteClass.UNKNOWN
        elif confidence < self.acceptance_confidence or reasons:
            reasons.append("requires_manual_review")
            status = DecisionStatus.REVIEW
            final_class = predicted_class
        else:
            reasons.append("thresholds_passed")
            status = DecisionStatus.ACCEPTED
            final_class = predicted_class

        subtype = None
        if final_class == WasteClass.METAL and request.magnetic is not None:
            subtype = "ferrous_candidate" if request.magnetic else "non_ferrous_candidate"

        return DecisionResult(
            final_class=final_class,
            confidence=round(confidence, 6),
            status=status,
            reason_codes=reasons,
            subtype=subtype,
        )
