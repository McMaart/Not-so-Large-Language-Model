import customtkinter as ctk


ctk.set_default_color_theme("dark-blue")
ctk.set_appearance_mode("system")


class GUI(ctk.CTk):
    def __init__(self, mvc_controller, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.title("Language Model Project")
        self.geometry("1600x800")
        self.mvc_controller = mvc_controller

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
        self.training = Training(container, self, mvc_controller)
        self.training.grid(row=0, column=0, sticky="nsew", columnspan=3)


class Training(ctk.CTkFrame):
    def __init__(self, parent, controller, mvc_controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.mvc_controller = mvc_controller

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
        #self.dataset_label = ctk.CTkLabel(self, text="Dataset")
        #self.dataset_label.grid(row=3, column=0)
        # Dropdown for choosing dataset
        self.selection = ctk.StringVar(value=self.datasets[0])
        self.dropdown = ctk.CTkOptionMenu(self, values=self.datasets, variable=self.selection)
        self.dropdown.grid(row=2, column=0)

        self.is_training = False
        # Buttons for starting/canceling training
        self.start_training_button = ctk.CTkButton(self, text="Start Training",
                                                   command=lambda:self.mvc_controller.start_training() if self.is_training == False
                                                   else self.mvc_controller.cancel_training())
        self.start_training_button.grid(row=3, column=0)
        #self.cancel_training = ctk.CTkButton(self, text="Cancel Training", command=self.cancel_training)
        #self.cancel_training.grid(row=5, column=0)

        # Field for displaying information
        self.training_info = ctk.CTkTextbox(self)
        self.training_info.grid(row=2, column=1, rowspan=2, sticky="nsew")

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

        # TODO function for returning available models
        self.models = ["Model 1", "Model 2", "Model 3"]

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
        self.chat.insert("end", f"User: {prompt}\n")
        self.entry.delete(0, "end")
        # TODO function for generating response
        response = "This functionality is coming soon..."  #Dummy
        self.chat.insert("end", f"{self.model_selection.get()}: {response}\n\n")
        self.history.append((prompt, response, self.model_selection.get()))
        self.evaluate()
        return

    def evaluate(self):
        # TODO function for evaluating response
        score = [(len(self.history[-1][0]) % 10) * 10, (len(self.history[-1][0]) % 8) * 10]  # dummy
        self.eval.insert("end", f"Prompt {len(self.history):>3} ({self.model_selection.get()}):\n")
        self.eval.insert("end", f"Metric 1: {score[0]:>3}")
        self.eval.insert("end", f" Metric 2: {score[1]:>3}")
        self.eval.insert("end", "\n\n")
        return


if __name__ == "__main__":
    import controller_mvc.py as c
    import model_1
    app = GUI(c.Controller(model_1))
    app.mainloop()
