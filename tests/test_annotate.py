from __future__ import annotations

import unittest
from io import BytesIO

from PIL import Image

from itb_harness.annotate import BoxAnnotation, annotate_boxes, annotate_click


class AnnotateClickTests(unittest.TestCase):
    def test_draws_marker_on_clicked_pixel(self) -> None:
        original = Image.new("RGB", (200, 120), (255, 255, 255))
        buffer = BytesIO()
        original.save(buffer, format="PNG")
        marked = annotate_click(buffer.getvalue(), 100, 60)

        image = Image.open(BytesIO(marked)).convert("RGB")
        self.assertEqual(image.size, (200, 120))
        self.assertNotEqual(image.getpixel((112, 60)), (255, 255, 255))

        red_pixels = 0
        for x in range(80, 121):
            for y in range(40, 81):
                r, g, b = image.getpixel((x, y))
                if r > 180 and g < 130 and b < 130:
                    red_pixels += 1
        self.assertGreater(red_pixels, 20)

    def test_annotate_boxes_draws_categories(self) -> None:
        original = Image.new("RGB", (120, 100), (0, 0, 0))
        buffer = BytesIO()
        original.save(buffer, format="PNG")
        marked = annotate_boxes(
            buffer.getvalue(),
            [
                BoxAnnotation(10, 10, 50, 40, "demon"),
                BoxAnnotation(60, 50, 110, 90, "spike trap"),
            ],
        )
        image = Image.open(BytesIO(marked)).convert("RGB")
        self.assertEqual(image.size, (120, 100))
        # The demon outline is magenta-ish.
        r, g, b = image.getpixel((10, 20))
        self.assertGreater(r, 120)
        self.assertLess(g, 150)
        self.assertGreater(b, 150)

    def test_clamps_marker_to_image(self) -> None:
        original = Image.new("RGB", (50, 50), (0, 0, 0))
        buffer = BytesIO()
        original.save(buffer, format="PNG")
        marked = annotate_click(buffer.getvalue(), 1000, -20)
        image = Image.open(BytesIO(marked))
        self.assertEqual(image.size, (50, 50))


if __name__ == "__main__":
    unittest.main()
