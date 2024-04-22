import unittest
from unittest.mock import patch, MagicMock
import gui

class TestGUIComponents(unittest.TestCase):


    def setUp(self):
        self.parent = MagicMock()
        #self.controller = MagicMock()
        controller = gui.GUI()

        self.training_instance = gui.Training(controller, controller)
        self.interaction_instance = gui.Interaction(controller, controller)

    def test_gui_initialization(self):
        controller = gui.GUI()
        """Test GUI component is initialized correctly."""
        self.assertIsNotNone(controller)
        self.assertTrue(getattr(controller, "title"))
        self.assertTrue(getattr(controller, "geometry"))
        self.assertTrue(getattr(controller, "training"))
        self.assertTrue(getattr(controller, "interaction"))



    def test_training_initialization(self):
        """Test Trainingcomponent is initialized correctly."""
        self.assertIsNotNone(self.training_instance)
        self.assertTrue(getattr(self.training_instance, "controller"))
        self.assertTrue(getattr(self.training_instance, "datasets"))
        self.assertTrue(getattr(self.training_instance, "label"))

        self.assertTrue(getattr(self.training_instance, "dataset_label"))

        self.assertTrue(getattr(self.training_instance, "selection"))
        self.assertTrue(getattr(self.training_instance, "dropdown"))

        self.assertTrue(getattr(self.training_instance, "training_info"))

        self.assertTrue(getattr(self.training_instance, "start_training"))
        self.assertTrue(getattr(self.training_instance, "cancel_training"))

    def test_interaction_initialization(self):
        """Test Interaction component is initialized correctly."""
        self.assertIsNotNone(self.interaction_instance)
        self.assertTrue(getattr(self.interaction_instance, "controller"))

        self.assertTrue(getattr(self.interaction_instance, "models"))

        #self.assertTrue(getattr(self.interaction_instance, "history"))

        self.assertTrue(getattr(self.interaction_instance, "model_label"))
        self.assertTrue(getattr(self.interaction_instance, "model_selection"))
        self.assertTrue(getattr(self.interaction_instance, "dropdown"))

        self.assertTrue(getattr(self.interaction_instance, "chat"))

        self.assertTrue(getattr(self.interaction_instance, "entry"))

        self.assertTrue(getattr(self.interaction_instance, "send"))

        self.assertTrue(getattr(self.interaction_instance, "eval_label"))
        self.assertTrue(getattr(self.interaction_instance, "eval"))

if __name__ == '__main__':
    unittest.main()