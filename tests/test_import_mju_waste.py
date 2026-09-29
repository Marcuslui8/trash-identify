from pathlib import Path

from scripts.import_mju_waste import collect_split_names, fallback_split, parse_boxes


def test_collect_split_names(tmp_path: Path) -> None:
    split = tmp_path / "train.json"
    split.write_text('["JPEGImages/0001.jpg", {"file": "0002.xml"}]', encoding="utf-8")
    assert collect_split_names(split) == {"0001", "0002"}


def test_parse_boxes(tmp_path: Path) -> None:
    annotation = tmp_path / "sample.xml"
    annotation.write_text(
        """<annotation><object><name>waste</name><bndbox>
        <xmin>10</xmin><ymin>20</ymin><xmax>90</xmax><ymax>80</ymax>
        </bndbox></object></annotation>""",
        encoding="utf-8",
    )
    assert parse_boxes(annotation, 100, 100) == [(10.0, 20.0, 90.0, 80.0)]


def test_fallback_split_is_stable() -> None:
    assert fallback_split("image-001") == fallback_split("image-001")
