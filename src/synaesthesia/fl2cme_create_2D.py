import pandas as pd

# import matplotlib.pyplot as plt
# import numpy as np

if __name__ == "__main__":
    """
    This script is used to create labels from the fl2cme_vconf.csv file.
    """
    # mode = "simple_binary_labels"
    mode = "binary_max_m_hour_next_n_hours"
    # Define the number of hours to look ahead
    m_hour = 1
    n_hours = 24

    path = ".../labels/fl2cme_vconf.csv"

    df = pd.read_csv(path)

    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Determine the minimum and maximum timestamps
    min_time = df["start_time"].min()
    max_time = df["end_time"].max()

    # Adjust min_time to the nearest earlier 12-minute interval
    min_time_adjusted = min_time - pd.Timedelta(
        minutes=min_time.minute % 12,
        seconds=min_time.second,
        microseconds=min_time.microsecond,
    )

    # Create a datetime index with a cadence of 12 minutes
    time_index = pd.date_range(start=min_time_adjusted, end=max_time, freq="12min")


    if mode == "binary_max_m_hour_next_n_hours":
        # Initialize the flarelabel_timeseries DataFrame
        columns = ["flareclass", "flareclass_category"] + [f"+{i}h" for i in range(1, n_hours + 1)]
        flarelabel_timeseries = pd.DataFrame(index=time_index, columns=columns)
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0


        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            # Create a mask for the time range of the flare
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )

            # Assign the flare class to the flarelabel_timeseries DataFrame
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

        # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
        flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
            "flareclass"
        ].apply(
            lambda x: (1 if str(x).startswith("M") or str(x).startswith("X") else 0)
        )

        # Fill in the hourly labels
        for h in range(1, n_hours + 1):
            # Calculate the number of 12-minute intervals in `h` hours
            h_intervals = h * 60 // 12
            # For each timestamp, check if there's a flare within the next `h` hours
            flarelabel_timeseries[f"+{h}h"] = (
                flarelabel_timeseries["flareclass_category"]
                .rolling(window=h_intervals, min_periods=1)
                .max()
                .shift(-h_intervals)
            )

        # Drop the last rows where the future hours cannot be calculated
        flarelabel_timeseries.dropna(subset=[f"+{h}h" for h in range(1, n_hours + 1)], inplace=True)

        # Calculate percentage of each class category for +1h as an example
        total_rows = len(flarelabel_timeseries)
        percentage_C = (
            (flarelabel_timeseries["+1h"] == 1).sum() / total_rows * 100
        )

        print(f"Percentage of M/X-class flares in the next hour: {percentage_C:.2f}%")

        flarelabel_timeseries.to_csv(
            f"labels/flarelabel_timeseries_hourly_{n_hours}hours.csv"
        )





    if mode == "binary_max_n_hours":
        # Initialize the flarelabel_timeseries DataFrame
        flarelabel_timeseries = pd.DataFrame(
            index=time_index, columns=["flareclass", "flareclass_category"]
        )
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0

        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            # Create a mask for the time range of the flare
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )

            # Assign the flare class to the flarelabel_timeseries DataFrame
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

        # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
        flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
            "flareclass"
        ].apply(
            lambda x: (1 if str(x).startswith("M") or str(x).startswith("X") else 0)
        )

        n_intervals = n_hours * 60 // 12  # Number of 12-minute intervals in n hours

        # Compute the maximum value of the next n hours for each timestamp and replace the original column
        flarelabel_timeseries["flareclass_category"] = (
            flarelabel_timeseries["flareclass_category"]
            .rolling(window=n_intervals, min_periods=1)
            .max()
            .shift(-n_intervals)
        )

        # Drop the last rows where the next n hours cannot be calculated
        flarelabel_timeseries.dropna(subset=["flareclass_category"], inplace=True)

        # Calculate percentage of each class category
        total_rows = len(flarelabel_timeseries)
        percentage_C = (
            (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
        )

        print(f"Percentage of M/X-class flares: {percentage_C:.2f}%")

        flarelabel_timeseries.to_csv(
            f"/mnt/data/flare_labels/flarelabel_timeseries_binary_{n_hours}hourmax.csv"
        )

    if mode == "max_n_hours":
        """
        For each 12-minute timestep, a label of class C, M, or X is assigned
        based on the maximum GOES class in the next n hours. This class is assigned
        to the entire interval and added to the flarelabel_timeseries DataFrame.
        The class category, based on the first letter of the flare class (C=1, M=2, X=3),
        is also added. Output is saved to a csv.
        """
        # Initialize the flarelabel_timeseries DataFrame
        flarelabel_timeseries = pd.DataFrame(
            index=time_index, columns=["flareclass", "flareclass_category"]
        )
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0

        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            # Create a mask for the time range of the flare
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )

            # Assign the flare class to the flarelabel_timeseries DataFrame
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

        # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
        flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
            "flareclass"
        ].apply(
            lambda x: (
                1
                if str(x).startswith("C")
                else (
                    2
                    if str(x).startswith("M")
                    else (3 if str(x).startswith("X") else 0)
                )
            )
        )

        n_intervals = n_hours * 60 // 12  # Number of 12-minute intervals in n hours

        # Compute the maximum value of the next n hours for each timestamp and replace the original column
        flarelabel_timeseries["flareclass_category"] = (
            flarelabel_timeseries["flareclass_category"]
            .rolling(window=n_intervals, min_periods=1)
            .max()
            .shift(-n_intervals)
        )

        # Drop the last rows where the next n hours cannot be calculated
        flarelabel_timeseries.dropna(subset=["flareclass_category"], inplace=True)

        # Calculate percentage of each class category
        total_rows = len(flarelabel_timeseries)
        percentage_C = (
            (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
        )
        percentage_M = (
            (flarelabel_timeseries["flareclass_category"] == 2).sum() / total_rows * 100
        )
        percentage_X = (
            (flarelabel_timeseries["flareclass_category"] == 3).sum() / total_rows * 100
        )

        print(f"Percentage of C-class flares: {percentage_C:.2f}%")
        print(f"Percentage of M-class flares: {percentage_M:.2f}%")
        print(f"Percentage of X-class flares: {percentage_X:.2f}%")

        flarelabel_timeseries.to_csv(
            f"/mnt/data/flare_labels/flarelabel_timeseries_{n_hours}hourmax.csv"
        )

    if mode == "simple_binary_labels":
        """
        For each 12-minute timestep, a label of class C, M, or X is assigned
        based on the active flare's GOES class. This class is assigned to the entire interval
        and added to the flarelabel_timeseries DataFrame. The class category,
        based on the first letter of the flare class (C=1, M=2, X=3), is also added.
        Output is saved to a csv.
        """

        # Initialize the flarelabel_timeseries DataFrame
        flarelabel_timeseries = pd.DataFrame(
            index=time_index, columns=["flareclass", "flareclass_category"]
        )
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0

        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            # Create a mask for the time range of the flare
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )

            # Assign the flare class to the flarelabel_timeseries DataFrame
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

            # Assign the flare class category
            flareclass_str = str(row["goes_class"])
            if flareclass_str.startswith("C"):
                flarelabel_timeseries.loc[mask, "flareclass_category"] = 1
            elif flareclass_str.startswith("M"):
                flarelabel_timeseries.loc[mask, "flareclass_category"] = 2
            elif flareclass_str.startswith("X"):
                flarelabel_timeseries.loc[mask, "flareclass_category"] = 3

        # Calculate percentage of each class category
        total_rows = len(flarelabel_timeseries)
        percentage_C = (
            (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
        )
        percentage_M = (
            (flarelabel_timeseries["flareclass_category"] == 2).sum() / total_rows * 100
        )
        percentage_X = (
            (flarelabel_timeseries["flareclass_category"] == 3).sum() / total_rows * 100
        )
        breakpoint()
        print(f"Percentage of C-class flares: {percentage_C:.2f}%")
        print(f"Percentage of M-class flares: {percentage_M:.2f}%")
        print(f"Percentage of X-class flares: {percentage_X:.2f}%")

        # Save the flarelabel_timeseries DataFrame to a CSV file
        flarelabel_timeseries.to_csv(
            "/home/data/flare_labels/flarelabel_timeseries.csv"
        )
        flarelabel_timeseries.to_csv(
            "/home/hannahruedisser/2024-ESL-Vigil/tests/test_data/flarelabel_timeseries.csv"
        )

        print(flarelabel_timeseries)

    if mode == "expanded_binary_labels":
        """
        For each 12-minute timestep, a label of class Cn, Mn, or Xn is assigned
        based on the active flare's GOES class, where n is a number from 1 to 9.
        Output is saved to a csv.
        """

        # Initialize the flarelabel_timeseries DataFrame
        flarelabel_timeseries = pd.DataFrame(
            index=time_index, columns=["flareclass", "flareclass_category"]
        )
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0

        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            # Create a mask for the time range of the flare
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )

            # Assign the flare class to the flarelabel_timeseries DataFrame
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

            # Assign the flare class category
            flareclass_str = str(row["goes_class"])
            class_letter = flareclass_str[0]
            class_number = int(flareclass_str[1])

            # Map the class letter to a base number and calculate the category
            base_numbers = {"C": 0, "M": 9, "X": 18}
            if class_letter in base_numbers:
                # Combine all X flares into one category
                if class_letter == "X":
                    flarelabel_timeseries.loc[mask, "flareclass_category"] = 18
                else:
                    category_number = base_numbers[class_letter] + class_number
                    flarelabel_timeseries.loc[mask, "flareclass_category"] = (
                        category_number
                    )
            # breakpoint()

        # Calculate count and percentage of each class category
        total_rows = len(flarelabel_timeseries)
        categories = sorted(flarelabel_timeseries["flareclass_category"].unique())
        counts = (
            flarelabel_timeseries["flareclass_category"].value_counts().sort_index()
        )

        percentages = {
            category: (counts[category] / total_rows * 100) for category in categories
        }

        for category in categories:
            print(
                f"Category {category}: Count = {counts[category]}, Percentage = {percentages[category]:.2f}%"
            )

        # breakpoint()

        # Save the flarelabel_timeseries DataFrame to a CSV file
        flarelabel_timeseries.to_csv(
            "/home/data/flare_labels/flarelabel_timeseries_expanded.csv"
        )
        flarelabel_timeseries.to_csv(
            "/home/dominika_u_malinowska/2024-ESL-Vigil/plots/flarelabel_timeseries_expanded.csv"
        )
        print(flarelabel_timeseries)

    elif mode == "regression_labels":
        """
        a matrix with time to next flare in minutes, separately for each class of flares
        Rows - timestamps every 12 min
        Col - Class
        Fill with a time since last flare (of any class) in minutes
        So once a flare happens, the time to next flare is reset to 0
        """

        pass

    elif mode == "classification_labels":
        """
        A label
        """
        pass
