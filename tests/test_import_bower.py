from scripts.import_bower import (
    background_crop,
    clip_box,
    map_material,
    remove_contained_boxes,
    split_for_image,
    yolo_box,
)


def test_material_mapping_is_conservative() -> None:
    assert map_material("Glass - Coloured") == "glass"
    assert map_material("Metal") == "metal"
    assert map_material("Fiber - Paper") == "paper"
    assert map_material("Plastic - Hard") == "plastic"
    assert map_material("Composite") == "unknown"
    assert map_material(None) == "unknown"


def test_box_conversion_clips_coordinates() -> None:
    row = {"x1": -10, "y1": 20, "x2": 60, "y2": 80}
    box = clip_box(row, 100, 100)
    assert box == (0.0, 20.0, 60.0, 80.0)
    assert yolo_box(box, 100, 100) == (0.3, 0.5, 0.6, 0.6)


def test_background_crop_never_intersects_expanded_object() -> None:
    crop = background_crop(900, 900, [(300, 300, 600, 600)])
    assert crop is not None
    assert crop[2] <= 264 or crop[0] >= 636 or crop[3] <= 264 or crop[1] >= 636


def test_split_is_stable() -> None:
    assert split_for_image("same-image") == split_for_image("same-image")
    assert split_for_image("same-image") in {"train", "validation", "test"}


def test_contained_component_box_is_removed() -> None:
    outer = (10.0, 10.0, 90.0, 90.0)
    cap = (30.0, 10.0, 60.0, 25.0)
    separate = (0.0, 0.0, 8.0, 8.0)
    assert remove_contained_boxes([outer, cap, separate]) == [outer, separate]
