import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


if __name__ == "__main__":

    path = "/home/data/flare_labels/flarelabel_timeseries.csv"
    # Load the dataset
    dataset = pd.read_csv(path)

    # Define outer boundaries as datetime strings
    outer_boundaries = ["2010-11-13T00:24:00", "2018-12-07T11:48:00"]

    # Convert boundaries to datetime objects
    start_boundary = pd.to_datetime(outer_boundaries[0])
    end_boundary = pd.to_datetime(outer_boundaries[1])

    # Filter dataset between boundaries
    filtered_data = dataset[
        (pd.to_datetime(dataset["Unnamed: 0"]) >= start_boundary)
        & (pd.to_datetime(dataset["Unnamed: 0"]) <= end_boundary)
    ]

    # Create a new column based on flareclass starting with 'M' or 'X'
    filtered_data["starts_with_M_or_X"] = (
        filtered_data["flareclass"].str.startswith(("M", "X")).astype(int)
    )

    # Convert 'Unnamed: 0' to datetime column
    filtered_data["Unnamed: 0"] = pd.to_datetime(filtered_data["Unnamed: 0"])

    # Group by quarter and sum the starts_with_M_or_X column
    quarterly_sum = filtered_data.groupby(
        filtered_data["Unnamed: 0"].dt.to_period("Q")
    )["starts_with_M_or_X"].sum()

    # Convert quarterly_sum to a dictionary
    quarterly_sums = quarterly_sum.to_dict()

    # Convert dictionary to a list of tuples for sorting
    quarterly_list = list(quarterly_sums.items())

    # Sort quarterly sums based on values (sums)
    quarterly_list_sorted = sorted(quarterly_list, key=lambda x: x[1], reverse=True)

    # Determine the number of quarterly sums in each group
    total_events = sum([item[1] for item in quarterly_list_sorted])
    num_quarters = len(quarterly_list_sorted)

    # Calculate the target sums for each group
    target_sum_group1 = 0.8 * total_events
    target_sum_group2 = 0.1 * total_events
    target_sum_group3 = 0.1 * total_events

    # Initialize groups
    group1 = []
    group2 = []
    group3 = []

    # Distribute quarterly sums into groups while maintaining the ratio of total events
    current_sum_group1 = 0
    current_sum_group2 = 0
    current_sum_group3 = 0

    for quarter, value in quarterly_list_sorted:
        if current_sum_group1 + value <= target_sum_group1:
            group1.append((quarter, value))
            current_sum_group1 += value
        elif current_sum_group2 + value <= target_sum_group2:
            group2.append((quarter, value))
            current_sum_group2 += value
        else:
            group3.append((quarter, value))
            current_sum_group3 += value

    # Function to get boundaries for a given quarter as a list of lists
    def get_boundaries(quarter):
        start_date = quarter.start_time.strftime("%Y%m%dT%H%M%S")
        end_date = (quarter.end_time - pd.Timedelta(seconds=60 * 60 * 24 - 1)).strftime(
            "%Y%m%dT%H%M%S"
        )  # Subtract 1 second
        return [[start_date, end_date]]

    # Function to print group boundaries
    def print_group_boundaries(group):
        boundaries = []
        for quarter, _ in group:
            boundaries.extend(get_boundaries(quarter))
        print(boundaries)

    # Print the results with boundaries as list of lists
    print("Group 1 (80%):")
    print_group_boundaries(group1)
    print("\nGroup 2 (10%):")
    print_group_boundaries(group2)
    print("\nGroup 3 (10%):")
    print_group_boundaries(group3)

    # Function to plot the timeline with color-coded groups using hlines
    def plot_timeline(groups):
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each group's boundaries with different colors
        colors = ["tab:blue", "tab:orange", "tab:green"]
        legend_labels = ["Group 1 (80%)", "Group 2 (10%)", "Group 3 (10%)"]
        legend_lines = []

        for idx, group in enumerate(groups):
            for quarter, _ in group:
                start_date = quarter.start_time
                end_date = quarter.end_time
                ax.hlines(idx + 1, start_date, end_date, color=colors[idx], linewidth=2)
            legend_lines.append(
                Line2D(
                    [0], [0], color=colors[idx], linewidth=2, label=legend_labels[idx]
                )
            )

        # Set labels and title
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Group 1", "Group 2", "Group 3"])
        ax.set_xlabel("Time")
        ax.set_title("Timeline with Color-Coded Groups")

        # Add legend
        ax.legend(handles=legend_lines)

        # Show plot
        plt.tight_layout()

        # Show plot
        plt.savefig("plots/timeline.png")

    # Plot the timeline with color-coded groups
    plot_timeline([group1, group2, group3])
