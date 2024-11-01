from threading import Thread
import customtkinter as ctk
import datetime
import training as m1
import glob
import os
from io_utils import prompt_model
from evaluation.uhhgpt_selenium import webdriver_setup, get_ratings

ctk.set_default_color_theme("dark-blue")
ctk.set_appearance_mode("system")


class GUI(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.title("Language Model Project")
        self.geometry("1600x800")

        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True, anchor="n")
        container.grid_rowconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)
        container.grid_columnconfigure(2, weight=1)

        # Interaction frame
        self.interaction = Interaction(container, self)
        self.interaction.grid(row=1, column=0, sticky="nsew", columnspan=3)

        # Training frame
        self.training = Training(container, self)
        self.training.grid(row=0, column=0, sticky="nsew", columnspan=3)


class Training(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        # TODO function that returns available datasets
        self.datasets = ["TinyStories 1", "TinyStories 2", "TinyStories 3"]  # dummy
        self.models = ["Model 1", "Model 2", "Model 3"]

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=7)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)
        #self.grid_rowconfigure(4, weight=0)
        #self.grid_rowconfigure(5, weight=0)
        #self.grid_rowconfigure(6, weight=0)

        # Label for training
        self.label = ctk.CTkLabel(self, text="Training")
        self.label.grid(row=0, column=0, columnspan=1)

        #self.model_label = ctk.CTkLabel(self, width=1, height=1, text="Model")
        #self.model_label.grid(row=1, column=0, rowspan=1)
        self.model_selection = ctk.StringVar(value=self.models[0])
        self.dropdown = ctk.CTkOptionMenu(self, values=self.models, variable=self.model_selection)
        self.dropdown.grid(row=1, column=0)

        # Label for dataset selection
        # self.dataset_label = ctk.CTkLabel(self, text="Dataset")
        # self.dataset_label.grid(row=3, column=0)
        # Dropdown for choosing dataset
        self.selection = ctk.StringVar(value=self.datasets[0])
        self.dropdown = ctk.CTkOptionMenu(self, values=self.datasets, variable=self.selection)
        self.dropdown.grid(row=2, column=0)

        self.is_training = False
        self.flag_list = [self.is_training]
        self.start_training_button = ctk.CTkButton(self, text="Start Training",
                                                   command=lambda: self.start_training() if self.is_training is False
                                                   else self.cancel_training())
        self.start_training_button.grid(row=3, column=0)

        # Field for displaying information
        self.training_info = ctk.CTkTextbox(self)
        self.training_info.grid(row=2, column=1, rowspan=2, sticky="nsew")

    def start_training(self):
        start = datetime.datetime.now()
        self.is_training = True
        self.flag_list[0] = True
        # self.start_training_button.configure(text="Cancel Training")
        self.training_info.insert("end",
                                  f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        training_thread = Thread(target=self.run_model)
        training_thread.start()

        self.training_info.insert("end", f"Model saved at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        end = datetime.datetime.now()
        self.training_info.insert("end", "Training time: " + str(end - start) + "\n\n")
        self.start_training_button.configure(text="Start Training")
        return

    def cancel_training(self):
        #TODO stop training
        self.training_info.insert("end", "Training canceled\n")
        self.is_training = False
        self.flag_list[0] = False
        self.start_training_button.configure(text="Start Training")
        return

    def run_model(self):
        self.start_training_button.configure(text="Cancel Training")
        # TODO function for training with different data sets (tuple with 9 cases ("Model 1" TinyStories 1"...)
        print(self.model_selection.get())
        match self.model_selection.get():
            case "Model 1":
                self.training_info.insert("end", "MODEL 1!\n\n")
                t, avg, len, l = m1.training_setup(flag_list=self.flag_list)
                # self.training_info.insert("end", f"{l}")
                self.training_info.insert("end", f"\nModel 1 Training time: {t:.5}s ({t / len:.4}s per batch)\n")
                self.training_info.insert("end", f"Average Loss: {avg:.5}\n")
                self.training_info.insert("end", f"Average Loss: {avg:.5f}\n\n")
            case "Model 2":
                self.training_info.insert("end", "MODEL 2 not implemented yet!\n\n")
            case "Model 3":
                self.training_info.insert("end", "MODEL 3 not implemented yet!\n\n")
            case _:
                self.training_info.insert("end", "No Model selected!\n\n")

    # def change_Button(self):
    # self.start_training_button.configure(text="Cancel Training")


class Interaction(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=8)
        self.grid_rowconfigure(2, weight=2)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=4)
        self.grid_columnconfigure(4, weight=1)

        self.models = [os.path.splitext(os.path.basename(model))[0] for model in glob.glob(os.path.join(
            "../trained_models", "*.pth"))]
        # self.models = ["Model 1", "Model 2", "Model 3"]

        # Prompt and response history
        self.history = []

        # Dropdown for choosing model
        self.model_label = ctk.CTkLabel(self, text="Select model")
        self.model_label.grid(row=0, column=0)
        self.model_selection = ctk.StringVar(value=self.models[0])
        self.dropdown = ctk.CTkOptionMenu(self, values=self.models, variable=self.model_selection)
        self.dropdown.grid(row=1, column=0)

        # Chat history
        self.chat = ctk.CTkTextbox(self)
        self.chat.grid(row=1, column=1, sticky="nsew")

        # Chat entry
        self.entry = ctk.CTkEntry(self, placeholder_text="Enter your prompt here")
        self.entry.grid(row=2, column=1, sticky="nsew")

        # Button for sending message
        self.send = ctk.CTkButton(self, text="→", command=self.submit_prompt)
        self.send.grid(row=2, column=1, sticky="e")

        # Response Evaluation
        self.eval_label = ctk.CTkLabel(self, text="Response Evaluation")
        self.eval_label.grid(row=0, column=3)
        self.eval = ctk.CTkTextbox(self)
        self.eval.grid(row=1, column=3, sticky="nsew")

    def return_model(self):
        return self.model_selection

    def submit_prompt(self):
        prompt = self.entry.get()
        self.chat.insert("end", f"Prompt {len(self.history) + 1}:\n")
        self.chat.insert("end", f"{prompt}\n")
        self.entry.delete(0, "end")
        response = prompt_model(self.model_selection.get(), prompt)
        self.chat.insert("end", f"{self.model_selection.get()}:\n {response}\n\n")
        self.history.append((prompt, response, self.model_selection.get()))
        self.evaluate()
        return

    def evaluate(self):
        # import this from somewhere in future
        eval_prompt = """
        The following story is a story written for children at the age of around 5 years old.
        Your task is to rate the story objectively.

        Story:
        {}

        Please rate the story in the following categories:
        - GRAMMAR (exclusively rate the grammar of the story)
        - SPELLING (exclusively rate the spelling of the story)
        - CONSISTENCY (exclusively rate the consistency of the story)
        - STORY (exclusively rate the story/plot of the story)
        - CREATIVITY (exclusively rate the creativity of the story)
        - STYLE (exclusively rate the linguistic style of the story)

        Rate objectively and rate each category independently from the others.
        Rate each category with a score from 1 to 10, with 1 being the worst and 10 being the best.

        It is crucial that your response ends with the following format (substitute <YOUR_CATEGORY_SCORE> with the score you gave for the respective category) and does not include any other text afterwords:

        GRAMMAR: <YOUR_GRAMMAR_SCORE>
        SPELLING: <YOUR_SPELLING_SCORE>
        CONSISTENCY: <YOUR_CONSISTENCY_SCORE>
        STORY: <YOUR_STORY_SCORE>
        CREATIVITY: <YOUR_CREATIVITY_SCORE>
        STYLE: <YOUR_STYLE_SCORE>
        """
        from prompt_testing.parse_model_reply import categories
        if not hasattr(self, 'driver') or self.driver is None:
            self.driver = webdriver_setup()
        ratings = get_ratings(self.driver, eval_prompt, [self.history[-1][1]])[-1]
        for cat, score in zip(categories.values(), ratings):
            self.eval.insert("end", f"{cat}: {score}\n")
        self.eval.insert("end", "\n")
        return


if __name__ == "__main__":
    app = GUI()
    app.mainloop()
