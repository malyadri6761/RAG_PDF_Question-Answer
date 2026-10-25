from app.ingestion.image_loader import image_to_segments

from .conftest import make_text_image, needs_tesseract


@needs_tesseract
def test_ocr_reads_text():
    segs = image_to_segments(make_text_image(["Invoice number 7731"]), "inv.png", "d1")
    assert segs[0].content_type == "image_ocr"
    assert "7731" in segs[0].text


def test_tiny_images_are_skipped():
    assert image_to_segments(make_text_image(["x"], size=(30, 30), font_size=10), "icon.png", "d1") == []


def test_describer_adds_caption():
    segs = image_to_segments(make_text_image(["chart"]), "c.png", "d1", page=3,
                             describer=lambda b: "A bar chart.")
    caption = [s for s in segs if s.content_type == "image_caption"][0]
    assert caption.text == "A bar chart." and caption.page == 3


def test_broken_describer_does_not_crash():
    def broken(_):
        raise RuntimeError("API down")
    segs = image_to_segments(make_text_image(["chart"]), "c.png", "d1", describer=broken)
    assert all(s.content_type != "image_caption" for s in segs)