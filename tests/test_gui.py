import unittest
import gui

class TestGui(unittest.TestCase):
    def test_gui_exists(self):
        self.assertIsNotNone(gui.GUI)  # add assertion here
        self.assertIsNotNone(gui.Training)
        self.assertIsNotNone(gui.Interaction)
        #print(gui.GUI.children)

class TestTraining(unittest.TestCase):
    def test_start_training(self):
        pass
        #gui.Training.start_training()
        #self.assertTrue(is_training)

    def test_cancel_training(self):
        pass
        #gui.Training.cancel_training()
        #self.assertFalse(is_training)

class TestInteraction(unittest.TestCase):

    def test_attributes(self):
        #self.assertEqual(["Model 1", "Model 2", "Model 3"], )
        #print(gui.Interaction.children)
        pass

    def test_submit_prompt(self):
        pass

    def evaluate(self):
        pass


if __name__ == '__main__':
    unittest.main()
