from __future__ import annotations

import unittest

from itb_harness.input import InputEvent, StubInput, get_input_backend


class StubInputTests(unittest.TestCase):
    def test_records_click_and_hover_in_order(self) -> None:
        pointer = StubInput()
        pointer.click(10, 20)
        pointer.hover(30, 40)
        self.assertEqual(
            pointer.events,
            [InputEvent("click", 10, 20), InputEvent("hover", 30, 40)],
        )

    def test_records_keys_and_right_click(self) -> None:
        pointer = StubInput()
        pointer.key("r")
        pointer.key_down("ctrl")
        pointer.key_up("ctrl")
        pointer.right_click()
        self.assertEqual(
            pointer.events,
            [
                InputEvent("key", key="r"),
                InputEvent("key_down", key="ctrl"),
                InputEvent("key_up", key="ctrl"),
                InputEvent("right_click"),
            ],
        )

    def test_get_stub_backend(self) -> None:
        pointer = get_input_backend("stub")
        self.assertEqual(pointer.name, "stub")
        pointer.close()


if __name__ == "__main__":
    unittest.main()
