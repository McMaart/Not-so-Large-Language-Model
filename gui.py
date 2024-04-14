import customtkinter as ctk

ctk.set_default_color_theme("dark-blue")
ctk.set_appearance_mode("system")

class GUI(ctk.CTk):
    def __init__(self, *args, **kwargs):
        ctk.CTk.__init__(self, *args, **kwargs)
        self.title("Language Model Project")
        self.geometry("1600x800")

        container = ctk.CTkFrame(self)
        container.pack(side="top",fill="both", expand=True, anchor="n")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

        self.training = Training(container, self)

        self.training.grid(row=0, column=0, sticky="nsew", columnspan=2)



class Training(ctk.CTkFrame):
    def __init__(self, parent, controller):
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller

        # TODO function that returns available datasets
        self.datasets = ["TinyStories 1", "TinyStories 2", "TinyStories 3"] # dummy

        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=7)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=0)


        # Label for training
        self.label = ctk.CTkLabel(self, text="Training")
        self.label.grid(row=0, column=0, columnspan=2)

        # Dropdown for choosing dataset
        self.selection = ctk.StringVar(value=self.datasets[0])
        self.dropdown = ctk.CTkOptionMenu(self, values=self.datasets, variable=self.selection)
        self.dropdown.grid(row=1, column=0)


        # Field for displaying information
        self.training_info = ctk.CTkTextbox(self)
        self.training_info.grid(row=2, column=1, rowspan=2, sticky="nsew")

        # Buttons for starting/canceling training
        self.start_training = ctk.CTkButton(self, text="Start Training", command=self.start_training)
        self.start_training.grid(row=2, column=0)
        self.cancel_training = ctk.CTkButton(self, text="Cancel Training", command=self.cancel_training)
        self.cancel_training.grid(row=3, column=0)
    
    def start_training(self):
        self.training_info.insert("end", "Training started\n")
        self.training_info.insert("end", "Model saved\n")
        return
    def cancel_training(self):
        self.training_info.insert("end", "Training canceled\n")
        return

app = GUI()
app.mainloop()