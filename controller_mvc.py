from threading import Thread
import gui_mvc
import training
import model_1
import datetime
from io_utils import get_vocabulary_idx, map_story_to_tensor, load_tiny_stories, clean_stories
from torchtext.data.utils import get_tokenizer
from time import perf_counter
import torch


class Controller:
    def __init__(self, model):
        self.model = model
        self.view = gui_mvc.GUI(self)

    def setup_callbacks(self):
        pass

    def start_training(self):
        start = datetime.datetime.now()
        self.view.training.is_training = True

        training_thread = Thread(target=self.run_model)
        training_thread.start()
        #self.run_model()

        self.view.training.training_info.insert("end", f"Model saved at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        end = datetime.datetime.now()
        self.view.training.training_info.insert("end", "Training time: " + str(end - start) + "\n\n")
       #self.view.training.is_training = False
       # self.view.training.start_training_button.configure(text="Start Training")
        return

    def cancel_training(self):
        self.view.training.is_training = False
        self.view.training.training_info.insert("end", "Training canceled\n")
        self.view.training.start_training_button.configure(text="Start Training")
        return

    def run_model(self):
        self.view.training.start_training_button.configure(text="Cancel Training")
        # TODO function for training with different data sets (tuple with 9 cases ("Model 1" TinyStories 1"...)
        print(self.view.training.model_selection.get())
        match self.view.training.model_selection.get():

            case "Model 1":
                self.view.training.training_info.insert("end", "MODEL 1!\n\n")
                stories = load_tiny_stories(20000)
                stories = clean_stories(stories)
                vocabulary = get_vocabulary_idx(stories)
                vocabulary_rev = {k: v for v, k in vocabulary.items()}
                tokenizer = get_tokenizer('basic_english')

                # model = torch.load('trained_models/model.pth').to(device)
                model = model_1.TransformerModel(len(vocabulary)).to(device)
                loss_fn = training.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), learning_rate)

                train_data = [training.get_batch(stories, i, vocabulary, tokenizer) for i in range(19000)]
                t0 = perf_counter()
                avg_loss, batch_list = training.train(train_data, model, loss_fn, optimizer)
                t = perf_counter() - t0

                print(f"Average Loss: {avg_loss:.5}")
                print(f"Average Loss: {avg_loss:.5f}")
                # torch.save(model, 'trained_models/model.pth')
                self.view.training.training_info.insert("end",
                                                        f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                # t, avg, len, l = training.do_training()
                self.view.training.training_info.insert("end", f"{batch_list}")
                self.view.training.training_info.insert("end",
                                                        f"\nTraining time: {t:.5}s ({t / len(train_data):.4}s per batch)")
                self.view.training.training_info.insert("end", f"Average Loss: {avg_loss:.5}\n")
                self.view.training.training_info.insert("end", f"Average Loss: {avg_loss:.5f}\n\n")

            case "Model 2":
                self.view.training.training_info.insert("end", "MODEL 2 not implemented yet!\n\n")
            case "Model 3":
                self.view.training.training_info.insert("end", "MODEL 3 not implemented yet!\n\n")
            case _:
                self.view.training.training_info.insert("end", "No Model selected!\n\n")

        self.view.training.is_training = False
        self.view.training.start_training_button.configure(text="Start Training")

    def load_data_and_initialize(self):
        # Additional setup logic
        pass


if __name__ == '__main__':

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    learning_rate = 1e-3
    # batch_size = 16
    max_seq_len = 16

    c = Controller(training)
    # c.setup_callbacks()
    gui_thread = Thread(target=c.view.mainloop())
    gui_thread.start()
    #c.view.mainloop()