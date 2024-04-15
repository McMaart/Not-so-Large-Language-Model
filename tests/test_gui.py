import unittest
import gui

class TestGui(unittest.TestCase):
    def test_gui_exists(self):
        app = gui.GUI()
        self.assertIsNotNone(app)  # add assertion here


if __name__ == '__main__':
    unittest.main()
