## Project Sketch: Tiny Stories
1. Project Overview
  The "TinyStories" project aims to create an interactive platform that generates short children's stories using a vocabulary suited for a 3-4 year old child. This project will develop two different (small) Large Language models (LLM’s): one based on the Transformer architecture and another using recurrent neural networks (RNNs). 
  Both models will be implemented in PyTorch. The goal is to compare the effectiveness of these models in generating coherent and creative stories that adhere to the vocabulary limitations by using appropriate evaluation criteria.
  This project is based on: “TinyStories: How Small Can Language Models Be and Still Speak Coherent English?” (Ronen Eldan, Yuanzhi Li, Microsoft Research, 2023)

3. Objectives:
    1. Develop Two Language Models,
    2. Efficiency
    3. Interactive Platform
   
5. Technical Requirements
    1. Data: TinyStories dataset (size: 2.2M stories, # different word types:1500, 4500, 20000, length sentences: short), data set creation with GPT 4 and limited vocabulary.
       Preprocessing: Tokenization, vocabulary restriction, and conversion into suitable formats for model training.
    2. Models
      - Transformer: Simplified transformer architecture suitable for handling limited vocabulary and shorter text sequences efficiently.
      - RNN: Comparable (w.r.t. resources)
      - Different Configurations, Embeddings, data?
    3.  Software and Tools
      - Python as the primary programming language.
      - Pycharm as primary IDE used by developers.
      - Jupyter Notebook for quick model analysis
      - CustomTkinter for GUI
      - PyTorch for implementing machine learning models.
      - Git for version control, hosted on GitHub.
      - Venv environments for managing dependencies and ensuring consistency across development setups. Hint: Conda does not automatically work with customtkinter
6. Project Milestones
    1.	Project Setup and Data Preparation:
        Set up version control,
        Data collection and preprocessing,
        Initial model design and setup of development environments,
    2.	Model Development and Initial Training:
        Implementation and training of the transformer and RNN models,
        Set up basic user interface and testing,
    3.	Interface Development and Model Integration:
        Development of the full-featured user interface,
        Integration of the models with the application,
        Begin internal testing and iterations based on team feedback,

7. Evaluation Metrics
    1.	Model Performance: Runtime, Accuracy (syntactically, semantically valid), Creativity (new outputs or how new? e.g. Cosine distance to the closest example from training data)
    2.  Operational Metrics: Response times, system reliability, error rates

8. Potential Challenges
    1.	Model Bias and Sensitivity
    2.  Data Quality and Diversity: Building a dataset that is diverse and representative while adhering to vocabulary restrictions.

9. Resources: 3 developers, 
    Computation: “Runs on Laptop” (more specific technical details once the models are trained)
