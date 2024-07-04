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


# Function to retrieve WandB run data from multiple projects
def get_wandb_run_data(entity, project_names, run_ids=None, limit=None):
    api = wandb.Api()

    runs = []
    for project_name in project_names:
        project_runs = api.runs(f"{entity}/{project_name}")
        runs.extend(project_runs)

    # Convert to DataFrame for sorting
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
            'history': history,
            'num_parameters': summary.get('num_parameters', None),
            'training_time': summary.get('training_time', None),
            'eval_loss': summary.get('eval_loss', None),
            'created_at': run.created_at,  # Capture the creation time
            'project': run.project  # Add project name
        })

    run_df = pd.DataFrame(run_data)

    # Filter by run_ids if provided
    if run_ids:
        run_df = run_df[run_df['run_id'].isin(run_ids)]

    # Sort by creation time and select the last 'limit' runs
    if limit:
        run_df = run_df.sort_values(by='created_at', ascending=False).head(limit)

    return run_df


# Function to plot WandB run data as lines
def plot_wandb_data(run_df):
    plt.figure(figsize=(12, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color_map = {run_id: colors[i % len(colors)] for i, run_id in enumerate(run_df['run_id'])}

    for _, run in run_df.iterrows():
        history = run['history']
        history_df = pd.DataFrame(history)

        if '# batches' in history_df.columns and 'batch_loss' in history_df.columns:
            eval_loss = run['eval_loss']
            print(f"Plotting run {run['name']} from project {run['project']} with eval_loss {eval_loss}")  # Debug print
            plt.plot(history_df['# batches'], history_df['batch_loss'],
                     label=f"{run['name']} ({run['project']}) - Eval Loss: {eval_loss}", color=color_map[run['run_id']], linewidth=2)

            # Add hardcoded vertical line for the second epoch
            plt.axvline(x=26954, color='black', linestyle='--', linewidth=1)
        else:
            print(f"Skipping run {run['name']} from project {run['project']} due to missing data.")

    plt.xlabel('Number of Batches')
    plt.ylabel('Loss')
    plt.title('Batch Loss')
    plt.legend()
    plt.show()


# Function to plot WandB run data as scatter plot
def plot_wandb_scatter(run_df):
    # Debug: Print the first few rows of the DataFrame
    print("Scatter plot data:")
    print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(
        run_df['num_parameters'],
        run_df['training_time'],
        c=run_df['eval_loss'],
        cmap='RdYlGn_r',  # Reverse Red-Yellow-Green colormap
        edgecolor='k',
        alpha=0.7
    )
    plt.colorbar(scatter, label='Eval Loss')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Training Time')
    plt.title('Training Time vs. Number of Parameters (colored by Eval Loss)')
    plt.grid(True)
    plt.show()


# Function to list run IDs for specific projects
def list_run_ids(entity, project_names):
    api = wandb.Api()

    for project_name in project_names:
        runs = api.runs(f"{entity}/{project_name}")
        for run in runs:
            print(f"Project: {project_name}, Run ID: {run.id}, Name: {run.name}")


# Function to list all available projects
def list_projects(entity):
    api = wandb.Api()
    projects = api.projects(entity=entity)
    project_names = [project.name for project in projects]
    return project_names


# Main function to execute the data retrieval and plotting
def main():
    if not login_to_wandb():
        print("Please log in to your Weights and Biases account using `wandb login` command.")
        exit(1)

    api = wandb.Api()
    entity = api.default_entity

    # List available projects
    print("Available projects:")
    projects = list_projects(entity)
    for project in projects:
        print(f"- {project}")

    project_input = input(
        "Enter project names to include (comma-separated) or press Enter for default project: ").strip()
    if project_input:
        project_names = [name.strip() for name in project_input.split(',')]
    else:
        project_names = ['ml_llm_project']

    plot_choice = input("Enter 'line' to plot line chart or 'scatter' to plot scatter chart: ").strip().lower()

    if plot_choice == 'line':
        print("Available runs:")
        list_run_ids(entity, project_names)

        run_ids = input("Enter run IDs to plot (comma-separated) or press Enter to plot all: ").strip()
        if run_ids:
            run_ids = run_ids.split(',')
        else:
            run_ids = None
        run_df = get_wandb_run_data(entity, project_names, run_ids)
        plot_wandb_data(run_df)

    elif plot_choice == 'scatter':
        limit = input("Enter the number of last runs to plot (or press Enter for no limit): ").strip()
        if limit:
            limit = int(limit)
        else:
            limit = None
        run_df = get_wandb_run_data(entity, project_names, limit=limit)

        # Debug: Print the DataFrame before filtering
        print("Data before filtering:")
        print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

        # Filter out rows with missing or invalid data for scatter plot
        run_df = run_df.dropna(subset=['num_parameters', 'training_time', 'eval_loss'])

        # Debug: Print the DataFrame after filtering NaNs
        print("Data after filtering NaNs:")
        print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

        run_df = run_df[(run_df['num_parameters'] > 0) & (run_df['training_time'] > 0)]

        # Debug: Print the DataFrame after filtering invalid values
        print("Data after filtering invalid values:")
        print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

        plot_wandb_scatter(run_df)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
