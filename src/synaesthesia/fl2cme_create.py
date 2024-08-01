import pandas as pd

# import matplotlib.pyplot as plt
# import numpy as np

if __name__ == "__main__":
    """
    This script is used to create labels from the fl2cme_vconf.csv file.
    This should eventually create a dataframe saved as csv file with the
    following columns:

    TO DO

    Index:
    Pandas datetime index for the whole range of the catalog. Make sure it
    has the same cadence as the data (12 minutes). Make sure to correctly
    interpolate all values for the whole range of dates!


    Possible target variables:

    - flareclass:
    A value between 0 and 3, where 0 is no flare, 1 is C class, 2 is M class,
    and 3 is X class. This is a possible target variable we could predict.

    - flare_timeline:
    Alternative way of predicting a flare. This is a numerical value that
    represents the time until the next flare in multiples of the cadence.
    This could be a regression target.

    - CME_associated:
    A boolean value indicating if there is a flare with an associated CME.
    Think about how to deal with different levels of cme_valid_confidence.

    - CME parameters such as cme_vel, cme_width, ...

    - think about other possible target variables
    """
    # mode = "simple_binary_labels"
    mode = "binary_max_1_hour_next_n_hours"
    # Define the number of hours to look ahead
    n_hours = 24

    path = "/mnt/data/flare_labels/fl2cme_vconf.csv"

    df = pd.read_csv(flarelabelpath)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    # Filter the dataframe for goes_class starting with 'X' or 'M'
    filtered_df = df[df["goes_class"].str.startswith(("X", "M"))]

    # Define the start and end of the time range
    start_time = filtered_df["start_time"].min().floor("h")
    end_time = filtered_df["end_time"].max().ceil("h")

    # Create an hourly time series dataframe
    time_index = pd.date_range(start=start_time, end=end_time, freq='h')
    time_series_df = pd.DataFrame(index=time_index, columns=["flare_class", "flareclass_category", "fl_lon", "fl_lat"])


    # Fill the flare_class, flareclass_category, fl_lon, and fl_lat
    for current_time in time_series_df.index:
        # Calculate the time range for the next 24 hours
        future_24h = filtered_df[(filtered_df["start_time"] >= current_time) & 
                                (filtered_df["start_time"] < current_time + pd.Timedelta(hours=24))]
        
        # Check for ongoing events
        ongoing_event = filtered_df[(filtered_df["start_time"] <= current_time) & 
                                    (filtered_df["end_time"] > current_time)]
        
        # Determine the row with the maximum goes_class, prioritizing X over M
        if not future_24h.empty:
            x_class_events = future_24h[future_24h["goes_class"].str.startswith("X")]
            if not x_class_events.empty:
                closest_event_row = x_class_events.iloc[0]
            else:
                m_class_events = future_24h[future_24h["goes_class"].str.startswith("M")]
                if not m_class_events.empty:
                    closest_event_row = m_class_events.iloc[0]
                else:
                    closest_event_row = None
        else:
            closest_event_row = None
        
        # Set the flare_class, fl_lon, fl_lat, and flareclass_category
        if closest_event_row is not None:
            time_series_df.loc[current_time, "flare_class"] = closest_event_row["goes_class"]
            time_series_df.loc[current_time, "fl_lon"] = closest_event_row["fl_lon"]
            time_series_df.loc[current_time, "fl_lat"] = closest_event_row["fl_lat"]
            time_series_df.loc[current_time, "flareclass_category"] = 1
        elif not ongoing_event.empty:
            # If there's an ongoing event
            time_series_df.loc[current_time, "flare_class"] = ongoing_event.iloc[0]["goes_class"]
            time_series_df.loc[current_time, "fl_lon"] = ongoing_event.iloc[0]["fl_lon"]
            time_series_df.loc[current_time, "fl_lat"] = ongoing_event.iloc[0]["fl_lat"]
            time_series_df.loc[current_time, "flareclass_category"] = 1
        else:
            # No events found in the next 24 hours
            time_series_df.loc[current_time, "flare_class"] = ""
            time_series_df.loc[current_time, ["fl_lon", "fl_lat"]] = None
            time_series_df.loc[current_time, "flareclass_category"] = 0

    # Convert flareclass_category to integer
    time_series_df["flareclass_category"] = time_series_df["flareclass_category"].astype(int)

    print(time_series_df)

    # Drop the last rows where the next n hours cannot be calculated
    time_series_df.dropna(subset=["flareclass_category"], inplace=True)

    # Calculate percentage of each class category
    total_rows = len(time_series_df)
    percentage_C = (
        (time_series_df["flareclass_category"] == 1).sum() / total_rows * 100
    )

    print(f"Percentage of M/X-class flares: {percentage_C:.2f}%")

    time_series_df.to_csv(
        f"labels/flarelabel_timeseries_binary_hourly.csv"
    )

    # df = pd.read_csv(path)

    # df["start_time"] = pd.to_datetime(df["start_time"])
    # df["end_time"] = pd.to_datetime(df["end_time"])

    # # Determine the minimum and maximum timestamps
    # min_time = df["start_time"].min()
    # max_time = df["end_time"].max()

    # # Adjust min_time to the nearest earlier 12-minute interval
    # min_time_adjusted = min_time - pd.Timedelta(
    #     minutes=min_time.minute % 12,
    #     seconds=min_time.second,
    #     microseconds=min_time.microsecond,
    # )

    # # Create a datetime index with a cadence of 12 minutes
    # time_index = pd.date_range(start=min_time_adjusted, end=max_time, freq="12min")


    # if mode == "binary_max_n_hours":
    #     # Initialize the flarelabel_timeseries DataFrame
    #     flarelabel_timeseries = pd.DataFrame(
    #         index=time_index, columns=["flareclass", "flareclass_category"]
    #     )
    #     flarelabel_timeseries["flareclass"] = 0
    #     flarelabel_timeseries["flareclass_category"] = 0

    #     # Populate the flareclass and flareclass_category columns
    #     for i, row in df.iterrows():
    #         # Create a mask for the time range of the flare
    #         mask = (flarelabel_timeseries.index >= row["start_time"]) & (
    #             flarelabel_timeseries.index <= row["end_time"]
    #         )

    #         # Assign the flare class to the flarelabel_timeseries DataFrame
    #         flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

    #     # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
    #     flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
    #         "flareclass"
    #     ].apply(
    #         lambda x: (1 if str(x).startswith("M") or str(x).startswith("X") else 0)
    #     )

    #     n_intervals = n_hours * 60 // 12  # Number of 12-minute intervals in n hours

    #     # Compute the maximum value of the next n hours for each timestamp and replace the original column
    #     flarelabel_timeseries["flareclass_category"] = (
    #         flarelabel_timeseries["flareclass_category"]
    #         .rolling(window=n_intervals, min_periods=1)
    #         .max()
    #         .shift(-n_intervals)
    #     )

    #     # Drop the last rows where the next n hours cannot be calculated
    #     flarelabel_timeseries.dropna(subset=["flareclass_category"], inplace=True)

    #     # Calculate percentage of each class category
    #     total_rows = len(flarelabel_timeseries)
    #     percentage_C = (
    #         (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
    #     )

    #     print(f"Percentage of M/X-class flares: {percentage_C:.2f}%")

    #     flarelabel_timeseries.to_csv(
    #         f"/mnt/data/flare_labels/flarelabel_timeseries_binary_{n_hours}hourmax.csv"
    #     )





    # if mode == "binary_max_n_hours":
    #     # Initialize the flarelabel_timeseries DataFrame
    #     flarelabel_timeseries = pd.DataFrame(
    #         index=time_index, columns=["flareclass", "flareclass_category"]
    #     )
    #     flarelabel_timeseries["flareclass"] = 0
    #     flarelabel_timeseries["flareclass_category"] = 0

    #     # Populate the flareclass and flareclass_category columns
    #     for i, row in df.iterrows():
    #         # Create a mask for the time range of the flare
    #         mask = (flarelabel_timeseries.index >= row["start_time"]) & (
    #             flarelabel_timeseries.index <= row["end_time"]
    #         )

    #         # Assign the flare class to the flarelabel_timeseries DataFrame
    #         flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

    #     # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
    #     flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
    #         "flareclass"
    #     ].apply(
    #         lambda x: (1 if str(x).startswith("M") or str(x).startswith("X") else 0)
    #     )

    #     n_intervals = n_hours * 60 // 12  # Number of 12-minute intervals in n hours

    #     # Compute the maximum value of the next n hours for each timestamp and replace the original column
    #     flarelabel_timeseries["flareclass_category"] = (
    #         flarelabel_timeseries["flareclass_category"]
    #         .rolling(window=n_intervals, min_periods=1)
    #         .max()
    #         .shift(-n_intervals)
    #     )

    #     # Drop the last rows where the next n hours cannot be calculated
    #     flarelabel_timeseries.dropna(subset=["flareclass_category"], inplace=True)

    #     # Calculate percentage of each class category
    #     total_rows = len(flarelabel_timeseries)
    #     percentage_C = (
    #         (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
    #     )

    #     print(f"Percentage of M/X-class flares: {percentage_C:.2f}%")

    #     flarelabel_timeseries.to_csv(
    #         f"/mnt/data/flare_labels/flarelabel_timeseries_binary_{n_hours}hourmax.csv"
    #     )

    # if mode == "max_n_hours":
    #     """
    #     For each 12-minute timestep, a label of class C, M, or X is assigned
    #     based on the maximum GOES class in the next n hours. This class is assigned
    #     to the entire interval and added to the flarelabel_timeseries DataFrame.
    #     The class category, based on the first letter of the flare class (C=1, M=2, X=3),
    #     is also added. Output is saved to a csv.
    #     """
    #     # Initialize the flarelabel_timeseries DataFrame
    #     flarelabel_timeseries = pd.DataFrame(
    #         index=time_index, columns=["flareclass", "flareclass_category"]
    #     )
    #     flarelabel_timeseries["flareclass"] = 0
    #     flarelabel_timeseries["flareclass_category"] = 0

    #     # Populate the flareclass and flareclass_category columns
    #     for i, row in df.iterrows():
    #         # Create a mask for the time range of the flare
    #         mask = (flarelabel_timeseries.index >= row["start_time"]) & (
    #             flarelabel_timeseries.index <= row["end_time"]
    #         )

    #         # Assign the flare class to the flarelabel_timeseries DataFrame
    #         flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

    #     # Assign flare class category based on the first letter of the flare class (C=1, M=2, X=3)
    #     flarelabel_timeseries["flareclass_category"] = flarelabel_timeseries[
    #         "flareclass"
    #     ].apply(
    #         lambda x: (
    #             1
    #             if str(x).startswith("C")
    #             else (
    #                 2
    #                 if str(x).startswith("M")
    #                 else (3 if str(x).startswith("X") else 0)
    #             )
    #         )
    #     )

    #     n_intervals = n_hours * 60 // 12  # Number of 12-minute intervals in n hours

    #     # Compute the maximum value of the next n hours for each timestamp and replace the original column
    #     flarelabel_timeseries["flareclass_category"] = (
    #         flarelabel_timeseries["flareclass_category"]
    #         .rolling(window=n_intervals, min_periods=1)
    #         .max()
    #         .shift(-n_intervals)
    #     )

    #     # Drop the last rows where the next n hours cannot be calculated
    #     flarelabel_timeseries.dropna(subset=["flareclass_category"], inplace=True)

    #     # Calculate percentage of each class category
    #     total_rows = len(flarelabel_timeseries)
    #     percentage_C = (
    #         (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
    #     )
    #     percentage_M = (
    #         (flarelabel_timeseries["flareclass_category"] == 2).sum() / total_rows * 100
    #     )
    #     percentage_X = (
    #         (flarelabel_timeseries["flareclass_category"] == 3).sum() / total_rows * 100
    #     )

    #     print(f"Percentage of C-class flares: {percentage_C:.2f}%")
    #     print(f"Percentage of M-class flares: {percentage_M:.2f}%")
    #     print(f"Percentage of X-class flares: {percentage_X:.2f}%")

    #     flarelabel_timeseries.to_csv(
    #         f"/mnt/data/flare_labels/flarelabel_timeseries_{n_hours}hourmax.csv"
    #     )

    # if mode == "simple_binary_labels":
    #     """
    #     For each 12-minute timestep, a label of class C, M, or X is assigned
    #     based on the active flare's GOES class. This class is assigned to the entire interval
    #     and added to the flarelabel_timeseries DataFrame. The class category,
    #     based on the first letter of the flare class (C=1, M=2, X=3), is also added.
    #     Output is saved to a csv.
    #     """

    #     # Initialize the flarelabel_timeseries DataFrame
    #     flarelabel_timeseries = pd.DataFrame(
    #         index=time_index, columns=["flareclass", "flareclass_category"]
    #     )
    #     flarelabel_timeseries["flareclass"] = 0
    #     flarelabel_timeseries["flareclass_category"] = 0

    #     # Populate the flareclass and flareclass_category columns
    #     for i, row in df.iterrows():
    #         # Create a mask for the time range of the flare
    #         mask = (flarelabel_timeseries.index >= row["start_time"]) & (
    #             flarelabel_timeseries.index <= row["end_time"]
    #         )

    #         # Assign the flare class to the flarelabel_timeseries DataFrame
    #         flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

    #         # Assign the flare class category
    #         flareclass_str = str(row["goes_class"])
    #         if flareclass_str.startswith("C"):
    #             flarelabel_timeseries.loc[mask, "flareclass_category"] = 1
    #         elif flareclass_str.startswith("M"):
    #             flarelabel_timeseries.loc[mask, "flareclass_category"] = 2
    #         elif flareclass_str.startswith("X"):
    #             flarelabel_timeseries.loc[mask, "flareclass_category"] = 3

    #     # Calculate percentage of each class category
    #     total_rows = len(flarelabel_timeseries)
    #     percentage_C = (
    #         (flarelabel_timeseries["flareclass_category"] == 1).sum() / total_rows * 100
    #     )
    #     percentage_M = (
    #         (flarelabel_timeseries["flareclass_category"] == 2).sum() / total_rows * 100
    #     )
    #     percentage_X = (
    #         (flarelabel_timeseries["flareclass_category"] == 3).sum() / total_rows * 100
    #     )
    #     breakpoint()
    #     print(f"Percentage of C-class flares: {percentage_C:.2f}%")
    #     print(f"Percentage of M-class flares: {percentage_M:.2f}%")
    #     print(f"Percentage of X-class flares: {percentage_X:.2f}%")

    #     # Save the flarelabel_timeseries DataFrame to a CSV file
    #     flarelabel_timeseries.to_csv(
    #         "/home/data/flare_labels/flarelabel_timeseries.csv"
    #     )
    #     flarelabel_timeseries.to_csv(
    #         "/home/hannahruedisser/2024-ESL-Vigil/tests/test_data/flarelabel_timeseries.csv"
    #     )

    #     print(flarelabel_timeseries)

    # if mode == "expanded_binary_labels":
    #     """
    #     For each 12-minute timestep, a label of class Cn, Mn, or Xn is assigned
    #     based on the active flare's GOES class, where n is a number from 1 to 9.
    #     Output is saved to a csv.
    #     """

    #     # Initialize the flarelabel_timeseries DataFrame
    #     flarelabel_timeseries = pd.DataFrame(
    #         index=time_index, columns=["flareclass", "flareclass_category"]
    #     )
    #     flarelabel_timeseries["flareclass"] = 0
    #     flarelabel_timeseries["flareclass_category"] = 0

    #     # Populate the flareclass and flareclass_category columns
    #     for i, row in df.iterrows():
    #         # Create a mask for the time range of the flare
    #         mask = (flarelabel_timeseries.index >= row["start_time"]) & (
    #             flarelabel_timeseries.index <= row["end_time"]
    #         )

    #         # Assign the flare class to the flarelabel_timeseries DataFrame
    #         flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]

    #         # Assign the flare class category
    #         flareclass_str = str(row["goes_class"])
    #         class_letter = flareclass_str[0]
    #         class_number = int(flareclass_str[1])

    #         # Map the class letter to a base number and calculate the category
    #         base_numbers = {"C": 0, "M": 9, "X": 18}
    #         if class_letter in base_numbers:
    #             # Combine all X flares into one category
    #             if class_letter == "X":
    #                 flarelabel_timeseries.loc[mask, "flareclass_category"] = 18
    #             else:
    #                 category_number = base_numbers[class_letter] + class_number
    #                 flarelabel_timeseries.loc[mask, "flareclass_category"] = (
    #                     category_number
    #                 )
    #         # breakpoint()

    #     # Calculate count and percentage of each class category
    #     total_rows = len(flarelabel_timeseries)
    #     categories = sorted(flarelabel_timeseries["flareclass_category"].unique())
    #     counts = (
    #         flarelabel_timeseries["flareclass_category"].value_counts().sort_index()
    #     )

    #     percentages = {
    #         category: (counts[category] / total_rows * 100) for category in categories
    #     }

    #     for category in categories:
    #         print(
    #             f"Category {category}: Count = {counts[category]}, Percentage = {percentages[category]:.2f}%"
    #         )

    #     # breakpoint()

    #     # Save the flarelabel_timeseries DataFrame to a CSV file
    #     flarelabel_timeseries.to_csv(
    #         "/home/data/flare_labels/flarelabel_timeseries_expanded.csv"
    #     )
    #     flarelabel_timeseries.to_csv(
    #         "/home/dominika_u_malinowska/2024-ESL-Vigil/plots/flarelabel_timeseries_expanded.csv"
    #     )
    #     print(flarelabel_timeseries)

    # elif mode == "regression_labels":
    #     """
    #     a matrix with time to next flare in minutes, separately for each class of flares
    #     Rows - timestamps every 12 min
    #     Col - Class
    #     Fill with a time since last flare (of any class) in minutes
    #     So once a flare happens, the time to next flare is reset to 0
    #     """

    #     pass

    # elif mode == "classification_labels":
    #     """
    #     A label
    #     """
    #     pass
