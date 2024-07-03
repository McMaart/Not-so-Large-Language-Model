import wandb
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


# Function to log in to WandB with retries
def login_to_wandb(max_retries=3, delay=5):
    retries = 0
    while retries < max_retries:
        try:
            if wandb.login():
                return True
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


# Function to retrieve WandB run data
def get_wandb_run_data(entity, project_name, run_ids=None):
    api = wandb.Api()

    if run_ids:
        runs = [api.run(f"{entity}/{project_name}/{run_id}") for run_id in run_ids if
                run_id]  # Ensure run_id is not empty
    else:
        runs = api.runs(f"{entity}/{project_name}")

    run_data = []

    for run in runs:
        summary = run.summary._json_dict
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}
        history = run.history()  # Retrieve full history

        run_data.append({
            'run_id': run.id,
            'name': run.name,
            'summary': summary,
            'config': config,
            'history': history
        })

    return pd.DataFrame(run_data)


# Function to plot WandB run data
def plot_wandb_data(run_df):
    plt.figure(figsize=(12, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_map = {run_id: colors[i % len(colors)] for i, run_id in enumerate(run_df['run_id'])}

    for _, run in run_df.iterrows():
        history = run['history']
        history_df = pd.DataFrame(history)

        if '# batches' in history_df.columns and 'batch_loss' in history_df.columns:
            print(f"Plotting run {run['name']}")  # Debug print
            plt.plot(history_df['# batches'], history_df['batch_loss'],
                     label=run['name'], color=color_map[run['run_id']], linewidth=2)

            # Add hardcoded vertical line for the second epoch
            plt.axvline(x=26954, color='black', linestyle='--', linewidth=1)
            #plt.text(26953, plt.ylim()[1], '', color='black', rotation=90, verticalalignment='top')
        else:
            print(f"Skipping run {run['name']} due to missing data.")

    plt.xlabel('Number of Batches')
    plt.ylabel('Loss')
    plt.title('Batch Loss')
    plt.legend()
    plt.show()


# Function to list run IDs
def list_run_ids(entity, project_name):
    api = wandb.Api()

    runs = api.runs(f"{entity}/{project_name}")

    for run in runs:
        print(f"Run ID: {run.id}, Name: {run.name}")


# Main function to execute the data retrieval and plotting
def main():
    if not login_to_wandb():
        print("Please log in to your Weights and Biases account using `wandb login` command.")
        exit(1)

    project_name = 'ml_llm_project'  # Your project name

    # Get the entity of the logged-in user
    api = wandb.Api()
    entity = api.default_entity

    # List run IDs
    print("Available runs:")
    list_run_ids(entity, project_name)

    # Option to specify run IDs
    run_ids = input("Enter run IDs to plot (comma-separated) or press Enter to plot all: ").strip()
    if run_ids:
        run_ids = run_ids.split(',')
    else:
        run_ids = None

    run_df = get_wandb_run_data(entity, project_name, run_ids)
    plot_wandb_data(run_df)


if __name__ == "__main__":
    main()
