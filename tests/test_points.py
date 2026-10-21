from __future__ import annotations

import unittest

from itb_harness.points import (
    PointAction,
    PointParseError,
    parse_point_actions,
    point_to_pixels,
)


class PointParsingTests(unittest.TestCase):
    def test_json_array(self) -> None:
        actions = parse_point_actions(
            '[{"point_2d": [500, 250], "label": "click"},'
            ' {"point_2d": [0, 1000], "label": "hover"}]'
        )
        self.assertEqual(
            actions,
            [
                PointAction(500, 250, "click"),
                PointAction(0, 1000, "hover"),
            ],
        )

    def test_code_fence_and_prose(self) -> None:
        content = """Sure:
```json
[{"point_2d": [100, 200], "label": "click"}]
```"""
        self.assertEqual(
            parse_point_actions(content),
            [PointAction(100, 200, "click")],
        )

    def test_single_object(self) -> None:
        self.assertEqual(
            parse_point_actions('{"point_2d": [10, 20], "label": "hover"}'),
            [PointAction(10, 20, "hover")],
        )

    def test_python_style_and_unquoted_key(self) -> None:
        self.assertEqual(
            parse_point_actions("{point_2d: [30, 40], 'label': 'click'}"),
            [PointAction(30, 40, "click")],
        )

    def test_multiple_standalone_objects(self) -> None:
        self.assertEqual(
            parse_point_actions(
                '{"point_2d": [1, 2], "label": "click"}\n'
                '{"point_2d": [3, 4], "label": "hover"}'
            ),
            [PointAction(1, 2, "click"), PointAction(3, 4, "hover")],
        )

    def test_actions_wrapper(self) -> None:
        self.assertEqual(
            parse_point_actions(
                '{"actions": [{"point_2d": [5, 6], "label": "click"}]}'
            ),
            [PointAction(5, 6, "click")],
        )

    def test_empty_array(self) -> None:
        self.assertEqual(parse_point_actions("[]"), [])

    def test_rejects_bad_label(self) -> None:
        with self.assertRaises(PointParseError):
            parse_point_actions('[{"point_2d": [1, 2], "label": "drag"}]')

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(PointParseError):
            parse_point_actions('[{"point_2d": [1001, 2], "label": "click"}]')

    def test_rejects_bad_point(self) -> None:
        with self.assertRaises(PointParseError):
            parse_point_actions('[{"point_2d": [1], "label": "click"}]')

    def test_rejects_non_json(self) -> None:
        with self.assertRaises(PointParseError):
            parse_point_actions("I would click the mech.")


class PointScalingTests(unittest.TestCase):
    def test_corners(self) -> None:
        self.assertEqual(
            point_to_pixels(PointAction(0, 0, "click"), 101, 101), (0, 0)
        )
        self.assertEqual(
            point_to_pixels(PointAction(1000, 1000, "click"), 101, 101),
            (100, 100),
        )

    def test_center(self) -> None:
        self.assertEqual(
            point_to_pixels(PointAction(500, 500, "click"), 101, 101),
            (50, 50),
        )

    def test_non_square(self) -> None:
        self.assertEqual(
            point_to_pixels(PointAction(250, 750, "hover"), 1920, 1080),
            (480, 809),
        )


if __name__ == "__main__":
    unittest.main()
