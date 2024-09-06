import unittest
from unittest.mock import patch, MagicMock
from CosineSim_Rouge_for_models import (
    generate_story,
    calculate_cosine_similarity,
    calculate_rouge,
    evaluate_model,
    save_scores_to_file
)
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import torch


class TestCosineSimRougeForModels(unittest.TestCase):

    @patch('CosineSim_Rouge_for_models.prompt_model')
    def test_generate_story(self, mock_prompt_model):
        mock_prompt_model.return_value = "Generated story text"
        result = generate_story("Once upon a time", "mock_model")
        self.assertEqual(result, "Generated story text")
        mock_prompt_model.assert_called_once()

    def test_calculate_cosine_similarity(self):
        vectorizer = TfidfVectorizer()
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The fast brown fox leaps over the lazy dog"
        vectorizer.fit([text1, text2])
        result = calculate_cosine_similarity(text1, text2, vectorizer)
        self.assertTrue(isinstance(result, float))
        self.assertGreaterEqual(result, 0)
        self.assertLessEqual(result, 1)

    def test_calculate_rouge(self):
        reference = "The quick brown fox jumps over the lazy dog"
        generated = "The quick brown fox jumps over the lazy dog"
        result = calculate_rouge(reference, generated)
        self.assertIn('rouge1', result)
        self.assertIn('rouge2', result)
        self.assertIn('rougeL', result)
        self.assertTrue(all(isinstance(value.fmeasure, float) for value in result.values()))

    @patch('CosineSim_Rouge_for_models.generate_story')
    @patch('CosineSim_Rouge_for_models.calculate_cosine_similarity')
    @patch('CosineSim_Rouge_for_models.calculate_rouge')
    @patch('CosineSim_Rouge_for_models.TfidfVectorizer')
    def test_evaluate_model(self, mock_tfidf, mock_rouge, mock_cosine, mock_generate):
        mock_tfidf_instance = MagicMock()
        mock_tfidf.return_value = mock_tfidf_instance
        mock_cosine.return_value = 0.9
        mock_rouge.return_value = {'rouge1': MagicMock(fmeasure=0.8), 'rouge2': MagicMock(fmeasure=0.7),
                                   'rougeL': MagicMock(fmeasure=0.75)}
        mock_generate.return_value = "Generated story completion"

        prompts = ["Once upon a time"]
        completions = ["... they lived happily ever after."]
        dataset = ["Once upon a time ... they lived happily ever after."]

        avg_cosim, std_cosim, avg_rouge, std_rouge = evaluate_model("mock_model", prompts, completions, dataset)

        self.assertTrue(isinstance(avg_cosim, float))
        self.assertTrue(isinstance(std_cosim, float))
        self.assertTrue(isinstance(avg_rouge, dict))
        self.assertTrue(isinstance(std_rouge, dict))
        self.assertEqual(len(avg_rouge), 3)
        self.assertEqual(len(std_rouge), 3)

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_scores_to_file(self, mock_open):
        avg_cosim = 0.9
        std_cosim = 0.05
        avg_rouge = {'rouge1': 0.8, 'rouge2': 0.7, 'rougeL': 0.75}
        std_rouge = {'rouge1': 0.05, 'rouge2': 0.04, 'rougeL': 0.045}
        save_scores_to_file("mock_model", avg_cosim, std_cosim, avg_rouge, std_rouge, "default", 0.7, 10)

        mock_open().write.assert_any_call("Average Cosine Similarity: 0.9\n")
        mock_open().write.assert_any_call("  rouge1: 0.8\n")


if __name__ == '__main__':
    unittest.main()
