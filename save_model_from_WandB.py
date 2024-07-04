import wandb
import torch
import os
from time import sleep

# Ensure wandb login with retries
def login_to_wandb(max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            if wandb.login():
                api = wandb.Api()
                return api.default_entity
            else:
                raise Exception("Login failed")
        except Exception as e:
            print(f"WandB login failed with error: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {delay} seconds... ({retries}/{max_retries})")
                sleep(delay)
            else:
                print("Failed to log in after multiple attempts. Exiting...")
                exit(1)

# Function to save the best model from a project based on the number of parameters
def save_best_model(entity, project_name, filename):
    api = wandb.Api()
    try:
        runs = api.runs(f"{entity}/{project_name}")
    except ValueError as e:
        print(e)
        print(f"Project {project_name} does not exist. Skipping...")
        return

    best_run = None
    best_eval_loss = float('inf')

    for run in runs:
        summary = run.summary._json_dict
        eval_loss = summary.get('eval_loss', None)
        if eval_loss is not None and eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_run = run

    if best_run is not None:
        model_file = best_run.config.get('model_path', 'trained_models/last_model.pth')
        if os.path.exists(model_file):
            torch.save(torch.load(model_file), os.path.join('trained_models', filename))
            print(f"Best model from project {project_name} saved as {filename}")
        else:
            print(f"Model file {model_file} does not exist. Skipping...")
    else:
        print(f"No valid runs found for project {project_name}. Skipping...")

def main():
    entity = login_to_wandb()

    projects_and_files = {
        'ml_llm_project_1M': '1MWandB.pth',
        'ml_llm_project_5M': '5MWandB.pth',
        'ml_llm_project_10M': '10MWandB.pth',
        'ml_llm_project_15M': '15MWandB.pth',
        'ml_llm_project_30M': '30MWandB.pth'
    }

    for project_name, filename in projects_and_files.items():
        save_best_model(entity, project_name, filename)

if __name__ == "__main__":
    main()
