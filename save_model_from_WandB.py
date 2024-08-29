import wandb
import torch
import os
from time import sleep

# Ensure wandb login with retries
def login_to_wandb(max_retries=3, delay=5):
    """
    Log in to WandB with a specified number of retries.

    :param max_retries: Maximum number of retry attempts.
    :param delay: Delay between retries in seconds.

    :return: entity (str): The WandB entity (user or team) upon successful login.
    """
    retries = 0
    while retries < max_retries:
        try:
            if wandb.login():
                api = wandb.Api()
                return api.default_entity  # Return the default entity after successful login
            else:
                raise Exception("Login failed")
        except Exception as e:
            print(f"WandB login failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {delay} seconds... ({retries}/{max_retries})")
                sleep(delay)  # Wait before retrying
            else:
                print("Failed to log in after multiple attempts. Exiting...")
                exit(1)

# Function to save the best model from a project based on the number of parameters
def save_best_model(entity, project_name, filename):
    """
    Save the best model from a WandB project based on the evaluation loss.

    :param entity: The WandB entity (user or team).
    :param project_name: The name of the WandB project.
    :param filename: The filename to save the best model as.

    :return: None
    """
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project_name}")  # Retrieve all runs from the project
    except ValueError as e:
        print(e)
        print(f"Project {project_name} does not exist. Skipping...")
        return

    best_run = None
    best_eval_loss = float('inf')  # Initialize with infinity to find the minimum

    # Iterate through all runs to find the one with the lowest evaluation loss
    for run in runs:
        summary = run.summary._json_dict
        eval_loss = summary.get('eval_loss', None)  # Get the evaluation loss from the run's summary
        if eval_loss is not None and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_run = run  # Update the best run if a lower loss is found

    if best_run is not None:
        model_file = best_run.config.get('model_path', 'trained_models/last_model.pth')  # Default to 'last_model.pth'
        if os.path.exists(model_file):
            torch.save(torch.load(model_file), os.path.join('trained_models', filename))  # Save the model
            print(f"Best model from project {project_name} saved as {filename}")
        else:
            print(f"Model file {model_file} does not exist. Skipping...")
    else:
        print(f"No valid runs found for project {project_name}. Skipping...")

def main():
    """
    Main function to log in to WandB and save the best models from multiple projects.

    :return: None
    """
    entity = login_to_wandb()  # Log in to WandB

    projects_and_files = {
        'ml_llm_project_1M': '1MWandB.pth',
        'ml_llm_project_5M': '5MWandB.pth',
        'ml_llm_project_10M': '10MWandB.pth',
        'ml_llm_project_15M': '15MWandB.pth',
        'ml_llm_project_30M': '30MWandB.pth'
    }

    # Iterate through each project and save the best model
    for project_name, filename in projects_and_files.items():
        save_best_model(entity, project_name, filename)

if __name__ == "__main__":
    main()
