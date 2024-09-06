import wandb
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep


# Function to log in to WandB with retries
def login_to_wandb(max_retries=3, delay=5):
    """
    Attempt to log in to WandB with a specified number of retries.

    :param max_retries: The maximum number of login attempts.
    :param delay: The delay between retries in seconds.

    :return: True if login is successful, otherwise exits the program.
    """
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
    """
    Retrieve run data from multiple WandB projects.

    :param entity: The WandB entity (user or team).
    :param project_names: List of project names to retrieve data from.
    :param run_ids: Specific run IDs to filter the results (optional).
    :param limit: The maximum number of recent runs to retrieve (optional).

    :return: run_df (DataFrame): DataFrame containing the run data.
    """
    api = wandb.Api()

    runs = []
    for project_name in project_names:
        project_runs = api.runs(f"{entity}/{project_name}")  # Retrieve runs for each project
        runs.extend(project_runs)

    # Convert to DataFrame for sorting
    run_data = []
    for run in runs:
        summary = run.summary._json_dict  # Retrieve summary metrics
        config = {k: v for k, v in run.config.items() if not k.startswith('_')}  # Filter config keys
        history = run.history()  # Retrieve full history of metrics

        run_data.append({
            'run_id': run.id,
            'name': run.name,
            'summary': summary,
            'config': config,
            'history': history,
            'num_parameters': summary.get('num_parameters', None),  # Get number of parameters if available
            'training_time': summary.get('training_time', None),  # Get training time if available
            'eval_loss': summary.get('eval_loss', None),  # Get evaluation loss if available
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


# Function to format the training time
def format_training_time(seconds):
    """
    Format training time from seconds into a human-readable format.

    :param seconds: Training time in seconds.

    :return: formatted_time (str): Formatted training time string.
    """
    if seconds is None:
        return "N/A"
    hours, remainder = divmod(seconds, 3600)  # Calculate hours and remainder
    minutes, seconds = divmod(remainder, 60)  # Calculate minutes and seconds
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(minutes)}m {int(seconds)}s"


# Function to plot WandB run data as lines
def plot_wandb_data(run_df, plot_info):
    """
    Plot WandB run data as a line plot.

    :param run_df: DataFrame containing the run data.
    :param plot_info: Information to include in the plot labels (e.g., 'eval_loss', 'training_time').

    :return: None
    """
    plt.figure(figsize=(12, 6), dpi=300)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Define colors for different runs
    color_map = {run_id: colors[i % len(colors)] for i, run_id in enumerate(run_df['run_id'])}  # Assign colors to runs

    for _, run in run_df.iterrows():
        history = run['history']
        history_df = pd.DataFrame(history)

        if '# batches' in history_df.columns and 'batch_loss' in history_df.columns:  # Check for required columns
            eval_loss = run['eval_loss']
            training_time = run['training_time']
            label = f"{run['name']}"
            if 'eval_loss' in plot_info:
                label += f" - Eval Loss: {eval_loss:.2f}"
            if 'training_time' in plot_info:
                formatted_time = format_training_time(training_time)
                label += f" - Training Time: {formatted_time}"

            print(f"Plotting run {run['name']} from project {run['project']} with label {label}")  # Debug print
            plt.plot(history_df['# batches'], history_df['batch_loss'],
                     label=label, color=color_map[run['run_id']], linewidth=2)

        else:
            print(f"Skipping run {run['name']} from project {run['project']} due to missing data.")

    plt.axvline(x=26955, color='black', linestyle='--')  # Hardcoded line for second epoch
    plt.xlabel('Number of Batches')
    plt.ylabel('Loss')
    plt.title('Batch Loss')
    plt.legend()
    plt.show()


# Function to plot WandB run data as scatter plot
def plot_wandb_scatter(run_df):
    """
    Plot WandB run data as a scatter plot.

    :param run_df: DataFrame containing the run data.

    :return: None
    """
    # Debug: Print the first few rows of the DataFrame
    print("Scatter plot data:")
    print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

    plt.figure(figsize=(12, 6), dpi=300)

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


# Function to plot model size vs training time
def plot_model_training_time(run_df):
    """
    Plot model size vs training time as a scatter plot.

    :param run_df: DataFrame containing the run data.

    :return: None
    """
    plt.figure(figsize=(12, 6), dpi=300)

    # Filter out rows with missing or invalid data
    run_df = run_df.dropna(subset=['num_parameters', 'training_time'])
    run_df = run_df[(run_df['num_parameters'] > 0) & (run_df['training_time'] > 0)]

    plt.scatter(run_df['num_parameters'], run_df['training_time'], alpha=0.7, edgecolor='k')
    for _, run in run_df.iterrows():  # Annotate points with run names
        plt.text(run['num_parameters'], run['training_time'], f"{run['name']}", fontsize=8)

    plt.xlabel('Number of Parameters')
    plt.ylabel('Training Time')
    plt.title('Model Size vs Training Time')
    plt.grid(True)
    plt.show()


# Function to list run IDs for specific projects
def list_run_ids(entity, project_names):
    """
    List run IDs for specific WandB projects.

    :param entity: The WandB entity (user or team).
    :param project_names: List of project names to retrieve run IDs from.

    :return: None
    """
    api = wandb.Api()

    for project_name in project_names:
        runs = api.runs(f"{entity}/{project_name}")  # Retrieve runs for each project
        for run in runs:
            print(f"Project: {project_name}, Run ID: {run.id}, Name: {run.name}")


# Function to list all available projects
def list_projects(entity):
    """
    List all available projects for a WandB entity.

    :param entity: The WandB entity (user or team).

    :return: project_names (list): List of project names.
    """
    api = wandb.Api()
    projects = api.projects(entity=entity)  # Retrieve all projects for the entity
    project_names = [project.name for project in projects]  # Extract project names
    return project_names


# Main function to execute the data retrieval and plotting
def main():
    """
    Main function to log in to WandB, retrieve data, and generate plots based on user input.

    :return: None
    """
    if not login_to_wandb():  # Ensure successful login before proceeding
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
        project_names = [name.strip() for name in project_input.split(',')]  # Parse input project names
    else:
        project_names = ['ml_llm_project']  # Default project

    plot_choice = input(
        "Enter 'line' to plot line chart, 'scatter' to plot scatter chart, or 'model_time' to plot model training time: ").strip().lower()

    if plot_choice == 'line':
        plot_info = input(
            "Enter 'eval_loss', 'training_time', or 'both' to include in the plot labels: ").strip().lower()
        if plot_info not in ['eval_loss', 'training_time', 'both']:
            print("Invalid choice. Exiting.")
            exit(1)

        print("Fetching runs...")
        run_df = get_wandb_run_data(entity, project_names)

        print("Available runs:")
        for _, run in run_df.iterrows():
            print(f"Project: {run['project']}, Run ID: {run['run_id']}, Name: {run['name']}")

        run_ids = input("Enter run IDs to plot (comma-separated) or press Enter to plot all: ").strip()
        if run_ids:
            run_ids = run_ids.split(',')
            run_df = run_df[run_df['run_id'].isin(run_ids)]  # Filter DataFrame by run IDs

        plot_wandb_data(run_df, plot_info)

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

    elif plot_choice == 'model_time':
        print("Fetching runs...")
        run_df = get_wandb_run_data(entity, project_names)

        print("Available runs:")
        for _, run in run_df.iterrows():
            print(f"Project: {run['project']}, Run ID: {run['run_id']}, Name: {run['name']}")

        run_ids = input("Enter run IDs to plot (comma-separated) or press Enter to plot all: ").strip()
        if run_ids:
            run_ids = run_ids.split(',')
            run_df = run_df[run_df['run_id'].isin(run_ids)]  # Filter DataFrame by run IDs

        # Debug: Print the DataFrame before filtering
        print("Data before filtering:")
        print(run_df[['num_parameters', 'training_time', 'eval_loss']].head())

        plot_model_training_time(run_df)

    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
