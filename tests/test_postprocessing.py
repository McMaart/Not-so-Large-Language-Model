import sys
import unittest
from io_utils import load_vocabulary, TinyStories, tokens_to_story


class TestPostProcessing(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.vocabulary = load_vocabulary()
        self.vocabulary_rev = {v: k for k, v in self.vocabulary.items()}
        self.data = TinyStories(self.vocabulary, dataset_path="../data/TinyStoriesV2")
        self.stories = self.data.get_stories()
        self.unk_token = self.vocabulary['<unk>']
        # list of story indices where all tokens are in the vocabulary (i.e., there are no unknown tokens).
        self.stories_without_unk = [40029, 40030, 40032, 40036, 40042, 40045, 40051, 40054, 40058, 40063, 40065, 40070,
                                    40076, 40083, 40087, 40092, 40099, 40102, 40121, 40125, 40126, 40127, 40129, 40132,
                                    40133, 40134, 40146, 40148, 40149, 40152, 40160, 40165, 40166, 40169, 40176, 40178,
                                    40180, 40188, 40192, 40195, 40196, 40197, 40199, 40203, 40209, 40211, 40218, 40225,
                                    40230, 40233, 40235, 40236, 40237, 40238, 40240, 40247, 40250, 40257, 41738, 41740,
                                    41743, 41748, 41750, 41764, 41787, 41791, 41792, 41798, 41799, 41802, 41804, 41805,
                                    41807, 41817, 41820, 41824, 41828,]

    def test_postprocessing(self):
        for i in self.stories_without_unk:
            story_str = self.stories[i]
            input_tensor = self.data[i]
            if self.unk_token in input_tensor:
                print(f"Warning: Dataset or vocabulary has changed, the story with index {i} contains unknown tokens; "
                      "skipping this story...", file=sys.stderr)
                continue

            str_token_list = [self.vocabulary_rev[idx] for idx in input_tensor.tolist()[1:-1]]
            reconstructed_story = tokens_to_story(str_token_list)
            self.assertEqual(story_str, reconstructed_story)


if __name__ == '__main__':
    unittest.main()
