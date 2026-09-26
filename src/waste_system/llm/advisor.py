from __future__ import annotations

from waste_system.schemas import DecisionResult, DecisionStatus, WasteClass


CLASS_NAMES = {
    WasteClass.GLASS: "玻璃類",
    WasteClass.METAL: "鐵鋁類",
    WasteClass.PAPER: "紙類",
    WasteClass.PLASTIC: "塑膠類",
    WasteClass.UNKNOWN: "無法分類",
}

GENERAL_GUIDANCE = {
    WasteClass.GLASS: "請清空內容物並簡單清洗；破損玻璃應妥善包覆，依所在地規定處理。",
    WasteClass.METAL: "請清空內容物並簡單清洗；若含其他材質，請先拆分可拆卸部分。",
    WasteClass.PAPER: "請保持乾燥並移除明顯塑膠或金屬附件；受油污紙品可能不適合紙類回收。",
    WasteClass.PLASTIC: "請清空內容物並簡單清洗，再依塑膠標示及所在地規定分類。",
    WasteClass.UNKNOWN: "請勿直接投入一般回收分類；建議人工確認材質及所在地清運規定。",
}


class TemplateAdvisor:
    """Safe fallback until a grounded LLM/RAG provider is configured."""

    def advise(self, result: DecisionResult, region: str) -> str:
        if result.status == DecisionStatus.RESCAN:
            return "影像品質不足，請改善光線、保持鏡頭穩定後重新掃描。"
        if result.status == DecisionStatus.REVIEW:
            qualifier = "此結果需要人工確認。"
        elif result.status == DecisionStatus.UNKNOWN:
            qualifier = "系統目前無法可靠分類。"
        else:
            qualifier = "辨識結果已通過目前的信心與一致性門檻。"
        label = CLASS_NAMES[result.final_class]
        confidence = f"{result.confidence:.0%}"
        return (
            f"系統判定為{label}，模型信心值為 {confidence}。{qualifier}"
            f"{GENERAL_GUIDANCE[result.final_class]}實際投放前請以 {region} 當地最新回收規定為準。"
        )
