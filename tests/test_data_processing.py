import sys
import unittest
from io_utils import load_vocabulary, TinyStories, tokens_to_story


class TestDataProcessing(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.vocabulary = load_vocabulary()
        self.vocabulary_rev = {v: k for k, v in self.vocabulary.items()}
        self.data = TinyStories(self.vocabulary, dataset_path="../data/TinyStoriesV2")
        self.stories = self.data.get_stories()["text"]
        self.unk_token = self.vocabulary['<unk>']
        self.stories_without_unk = [28525, 28526, 28528, 28529, 28534, 28535, 28536, 28543, 28546, 28547, 28550, 28551,
                                    28552, 28553, 28555, 28556, 28559, 28561, 28580, 28582, 28583, 28589, 28597, 28599,
                                    28601, 28604, 28605, 28607, 28612, 28617, 28623, 28625, 28626, 28628, 28631, 28632,
                                    28636, 28638, 28640, 28650, 28652, 28657, 28662, 28672, 28673, 28676, 28684, 28685,
                                    28687, 28688, 28700, 28703, 28708, 28710, 28712, 28719, 28725, 28727, 28728, 28729,
                                    28734, 28736, 28738, 28741, 28743, 28746, 28747, 28749, 28752, 28753, 28762, 28763,
                                    28766, 28770, 28780, 28784, 28785]

    def test_postprocessing(self):
        for i in self.stories_without_unk:
            story_str = self.stories[i]
            input_tensor = self.data[i]
            if self.unk_token in input_tensor:
                print(f"Dataset or vocabulary has changed, the story with index {i} contains unknown tokens\n"
                      "Skipping this story...", file=sys.stderr)
                continue

            str_token_list = [self.vocabulary_rev[idx] for idx in input_tensor.tolist()[1:-1]]
            reconstructed_story = tokens_to_story(str_token_list)
            self.assertEqual(story_str, reconstructed_story)

if __name__ == '__main__':
    unittest.main()