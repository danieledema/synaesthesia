import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

    path = "/home/data/flare_labels/fl2cme_vconf.csv"
    user = "hannah"
    user = "d"

    if user == "hannah":
        df = pd.read_csv(path)

        df["start_time"] = pd.to_datetime(df["start_time"])
        df["end_time"] = pd.to_datetime(df["end_time"])

        # Determine the minimum and maximum timestamps
        min_time = df["start_time"].min()
        max_time = df["end_time"].max()


        breakpoint()
        
        # Adjust min_time to the nearest earlier 12-minute interval
        min_time_adjusted = min_time - pd.Timedelta(
            minutes=min_time.minute % 12,
            seconds=min_time.second,
            microseconds=min_time.microsecond,
        )

        # Create a datetime index with a cadence of 12 minutes
        time_index = pd.date_range(start=min_time_adjusted, end=max_time, freq="12min")

        # Initialize the flarelabel_timeseries DataFrame
        flarelabel_timeseries = pd.DataFrame(
            index=time_index, columns=["flareclass", "flareclass_category"]
        )
        flarelabel_timeseries["flareclass"] = 0
        flarelabel_timeseries["flareclass_category"] = 0

        # Populate the flareclass and flareclass_category columns
        for i, row in df.iterrows():
            mask = (flarelabel_timeseries.index >= row["start_time"]) & (
                flarelabel_timeseries.index <= row["end_time"]
            )
            flarelabel_timeseries.loc[mask, "flareclass"] = row["goes_class"]
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
        df.to_csv("/home/hannahruedisser/2024-ESL-Vigil/tests/test_data/fl2cme.csv")
        print(flarelabel_timeseries)

    else:

        # df = pd.read_csv(path)

        # print(df.columns)
        # print(df.head)

        # Index(['flare_id', 'start_time', 'peak_time', 'end_time', 'goes_class',
        #    'noaa_active_region', 'fl_lon', 'fl_lat', 'fl_loc_src', 'ssw_flare_id',
        #    'hinode_flare_id', 'primary_verified', 'secondary_verified',
        #    'candidate_ars', 'cme_id', 'fl_pa', 'cme_mpa', 'diff_a', 'cme_vel',
        #    'cme_width', 'cme_assoc_conf', 'cdaw_cme_id', 'cdaw_cme_width',
        #    'cdaw_cme_vel', 'cdaw_cme_pa', 'donki_cme_id', 'donki_cme_half_angle',
        #    'donki_cme_vel', 'lowcat_cme_id', 'lowcat_cme_width', 'lowcat_cme_vel',
        #    'lowcat_cme_pa', 'cme_valid_conf'],
        #   dtype='object')

        df_filtered = pd.read_csv(
            path,
            usecols=[
                "flare_id",
                "start_time",
                "peak_time",
                "end_time",
                "goes_class",
                "fl_lon",
                "fl_lat",
            ],
            dtype={
                "flare_id": "str",
                "goes_class": "str",
                "fl_lon": "float64",
                "fl_lat": "float64",
            },
            parse_dates=["start_time", "peak_time", "end_time"],
        )

        # Plot flare location
        plt.figure(figsize=(10, 10))  # Set the figure size for better readability
        plt.scatter(
            df_filtered["fl_lon"], df_filtered["fl_lat"], color="red", alpha=0.5, s=10
        )  # Plot scatter with fl_lon as x and fl_lat as y

        # Set limits for x and y axes
        plt.xlim(-90, 90)
        plt.ylim(-90, 90)

        # Draw a circle centered at (0, 0) with radius 90
        circle = plt.Circle((0, 0), 90, color="blue", fill=False)
        plt.gca().add_artist(circle)

        plt.title("Flare Locations")  # Title of the plot
        plt.xlabel("Longitude (fl_lon)")  # X-axis label
        plt.ylabel("Latitude (fl_lat)")  # Y-axis label
        plt.grid(True)  # Show grid
        plt.savefig("fl2cme_flare_location.jpg")  
      
        
        # Plot flare class vs flare time
        # Calculate the flare duration (difference between end_time and start_time)
        df_filtered['flare_duration'] = (df_filtered['end_time'] - df_filtered['start_time']).dt.total_seconds() / 60  # Convert to minutes

        # Map the goes_class to numerical values for plotting
        # This mapping depends on the specific format of goes_class. Here's an example mapping:
        class_mapping = {'C': 1, 'M': 2, 'X': 3}
        # Extract the first letter of each goes_class and map it
        df_filtered['goes_class_num'] = df_filtered['goes_class'].str[0].map(class_mapping)
        # Plot flare duration vs. flare class
        plt.figure(figsize=(10, 6))  # Set the figure size for better readability
        plt.scatter(df_filtered['flare_duration'], df_filtered['goes_class_num'], color="green", alpha=0.5, s=10)

        plt.title("Flare Duration vs. Flare Class")  # Title of the plot
        plt.xlabel("Flare Duration (minutes)")  # X-axis label
        plt.ylabel("Flare Class")  # Y-axis label
        plt.yticks(list(class_mapping.values()), list(class_mapping.keys()))  # Set y-ticks to flare classes
        # plt.grid(True)  # Show grid
        plt.savefig("fl2cme_flare_duration_vs_class.jpg")  # Save the plot

        # Assuming df_filtered is your DataFrame and 'goes_class' is the column with categorical data
        # Sort the unique values of 'goes_class' alphabetically
        unique_sorted = sorted(df_filtered['goes_class'].unique())

        # Create a mapping from the sorted unique labels to a numeric representation
        class_mapping = {value: key for key, value in enumerate(unique_sorted)}

        # Map the 'goes_class' column to its numeric representation using the sorted mapping
        df_filtered['goes_class_num'] = df_filtered['goes_class'].map(class_mapping)

        # Plot flare duration vs. flare class
        plt.figure(figsize=(10, 6))  # Set the figure size for better readability
        plt.scatter(df_filtered['flare_duration'], df_filtered['goes_class_num'], color="green", alpha=0.5, s=10)

        plt.title("Flare Duration vs. Flare Class")  # Title of the plot
        plt.xlabel("Flare Duration (minutes)")  # X-axis label
        plt.ylabel("Flare Class")  # Y-axis label
        
        n = 5  # For example, to plot every 5th label

        # Assuming class_mapping is already defined and contains your flare class mappings
        tick_values = list(class_mapping.values())
        tick_labels = list(class_mapping.keys())

        # Select every nth element for both values and labels
        selected_tick_values = tick_values[::n]
        selected_tick_labels = tick_labels[::n]
 
        plt.yticks(selected_tick_values, selected_tick_labels)  # Set y-ticks to every nth flare class

        # plt.yticks(list(class_mapping.values()), list(class_mapping.keys()))  # Set y-ticks to flare classes
        # plt.grid(True)  # Show grid
        plt.savefig("fl2cme_flare_duration_vs_class_detailed.jpg")  # Save the plot

        
        # Calculate statistics on flare duration
        # All classes
        fl_dur_all_class_stats = df_filtered.groupby("goes_class")["flare_duration"].describe()
        print("Statistics of flare duration by class:")
        print(fl_dur_all_class_stats)
        fl_dur_all_class_stats.to_csv('flare_duration_stats_all_class.csv', index=True)

        
        # Group so that only first character of a class is considered i.e. C, X, M
        df_filtered['goes_class_group1'] = df_filtered['goes_class'].str[:1]
        fl_dur_grouped_stats = df_filtered.groupby("goes_class_group1")["flare_duration"].describe()

        print("Statistics of flare duration by class:")
        print(fl_dur_grouped_stats)
        fl_dur_grouped_stats.to_csv('flare_duration_stats_1char_class.csv', index=True)

        # Group so that only first two characters of a class are considered e.g. C1, C2, etc.
        df_filtered['goes_class_group2'] = df_filtered['goes_class'].str[:2]
        fl_dur_grouped_stats = df_filtered.groupby("goes_class_group2")["flare_duration"].describe()

        print("Statistics of flare duration by class:")
        print(fl_dur_grouped_stats)
        fl_dur_grouped_stats.to_csv('flare_duration_stats_2char_class.csv', index=True)
        
        # Extract the 'mean' and 'std' column for plotting
        mean_flare_duration = fl_dur_grouped_stats['mean']
        std_flare_duration = fl_dur_grouped_stats['std']

                
        # Plotting
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.errorbar(mean_flare_duration.index, mean_flare_duration.values, yerr=std_flare_duration.values, fmt='o', capsize=5, linestyle='-', marker='s', markersize=5, label='Mean with STD')
        plt.xlabel('Flare Class')  # Set the x-axis label
        plt.ylabel('Mean Flare Duration (min)')  # Set the y-axis label
        plt.title('Mean Flare Duration by Flare Class')  # Set the title
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to not cut off labels
        plt.savefig("fl2cme_mean_flare_duration_vs_class.jpg")  # Save the plot




        # Plot time between flares
        # Calculate the time differences between the end of one flare and the start of the next
        # Shift the 'start_time' column up by one row to align the end of one flare with the start of the next
        df_filtered['next_start'] = df_filtered['start_time'].shift(-1)
        # Calculate the difference in min
        df_filtered['time_diff_h'] = (df_filtered['next_start'] - df_filtered['end_time']).dt.total_seconds() / 3660

        # Plot the time between flares
        plt.figure(figsize=(10, 6))
        plt.scatter(df_filtered['end_time'], df_filtered['time_diff_h'])
        plt.title('Time Between Flares')
        plt.xlabel('End Time of Flare')
        plt.ylabel('Time to Next Flare (h)')
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
        plt.savefig("fl2cme_flare_time_in_between.jpg")  # Save the plot
        

        
        # Calculate statistics on time between flares
        # All classes
        fl_diff_all_class_stats = df_filtered.groupby("goes_class")["time_diff_h"].describe()
        print("Statistics of time between flares by class:")
        print(fl_diff_all_class_stats)
        fl_diff_all_class_stats.to_csv('time_diff_h_stats_all_class.csv', index=True)

        
        # Group so that only first character of a class is considered i.e. C, X, M
        fl_diff_grouped_stats = df_filtered.groupby("goes_class_group1")["time_diff_h"].describe()

        print("Statistics of time between flares by class:")
        print(fl_diff_grouped_stats)
        fl_diff_grouped_stats.to_csv('time_diff_h_stats_1char_class.csv', index=True)

        # Group so that only first two characters of a class are considered e.g. C1, C2, etc.
        fl_diff_grouped_stats = df_filtered.groupby("goes_class_group2")["time_diff_h"].describe()

        print("Statistics of time between flares by class:")
        print(fl_diff_grouped_stats)
        fl_diff_grouped_stats.to_csv('time_diff_h_stats_2char_class.csv', index=True)
        

        # Extract the 'mean' and 'std' column for plotting
        mean_flare_diff = fl_diff_grouped_stats['mean']
        std_flare_diff = fl_diff_grouped_stats['std']

                
        # Plotting
        plt.figure(figsize=(10, 6))  # Set the figure size
        plt.errorbar(mean_flare_diff.index, mean_flare_diff.values, yerr=std_flare_diff.values, fmt='o', capsize=5, linestyle='-', marker='s', markersize=5, label='Mean with STD')
        plt.xlabel('Flare Class')  # Set the x-axis label
        plt.ylabel('Mean Difference in between Flares (h)')  # Set the y-axis label
        plt.title('Mean and STD of Difference in between Flares by Flare Class')  # Set the title
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.tight_layout()  # Adjust layout to not cut off labels
        plt.legend()  # Show legend
        plt.savefig("fl2cme_mean_flare_diff_vs_class.jpg")  # Save the plot


    
        # Define bins for the histogram
        bins = [float('-inf'), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, float('inf')]
        labels = ['overlaping','<1h', '1-2h', '2-3h', '3-4h', '4-5h', '5-6h', '6-7h', '7-8h', '8-9h', '9-10h', '>10h']

        # Categorize each time difference into bins
        df_filtered['time_diff_bin'] = pd.cut(df_filtered['time_diff_h'], bins=bins, labels=labels, right=False)

        # Count the occurrences in each bin
        time_diff_counts = df_filtered['time_diff_bin'].value_counts(sort=False)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.bar(time_diff_counts.index, time_diff_counts.values, color='skyblue')
        plt.xlabel('Time Difference (h)')
        plt.ylabel('Number of Entries')
        plt.title('Histogram of Time Difference between Flares Grouped by Hour')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("fl2cme_hist_flare_diff.jpg")  # Save the plot


        # Calculate the flare rate per year
        # Calculate the observation period in years
        min_time = df_filtered['start_time'].min()
        max_time = df_filtered['end_time'].max()
        total_years = (max_time - min_time).days / 365.25

        # Initialize an empty DataFrame for the results
        results_df = pd.DataFrame()

        # Calculate the rate for each category and store in the DataFrame
        for category in ['goes_class_group1', 'goes_class_group2']:
            flare_counts = df_filtered.groupby(category).size()
            flare_rates = flare_counts / total_years
            print(f"Average annual rate of flares per {category}:")
            print(flare_rates)

            # Convert flare_rates to DataFrame and rename the column
            flare_rates_df = flare_rates.to_frame(name='Flare Rate')
            # Optionally, rename the index
            flare_rates_df.index.rename('Flare Class', inplace=True)

            # Save the rates to a CSV file
            flare_rates_df.to_csv(f'flare_rate_{category}.csv', index=True)


        breakpoint()



        


        # Debuging the issue with lat-lon flare location (1997-2010)
        # path = "/home/data/flare_labels/fl2cme_soho.csv"

        # df_soho = pd.read_csv(path, usecols=['flare_id', 'start_time', 'peak_time', 'end_time', 'goes_class', 'fl_lon', 'fl_lat', 'eit_lon', 'eit_lat'])

        # # Fixing errors in csv (two values have additional .0 at the end so can't be convert to float)
        # # Define a function to check if a cell contains two dots
        # def contains_two_dots(x):
        #     if isinstance(x, str):
        #         return x.count('.') == 2
        #     return False

        # # Apply this function across the DataFrame and filter rows
        # rows_with_two_dots = df_soho[df_soho.applymap(contains_two_dots).any(axis=1)]

        # # Display the filtered rows
        # print(rows_with_two_dots)

        # # breakpoint()
        # df_soho.loc[135, 'eit_lon'] = df_soho.loc[135, 'eit_lon'][:-2]
        # df_soho.loc[2127, 'eit_lon'] = df_soho.loc[2127, 'eit_lon'][:-2]

        # df_soho['eit_lon'] = df_soho['eit_lon'].astype('float64')

        # # breakpoint()

        # plt.figure(figsize=(10, 10))  # Set the figure size for better readability
        # plt.scatter(df_soho['fl_lon'], df_soho['fl_lat'], color='red', alpha=0.5, s=10)  # Plot scatter with fl_lon as x and fl_lat as y
        # plt.scatter(df_soho['eit_lon'], df_soho['eit_lat'], color='blue', alpha=0.5, s=10)  # Plot scatter with fl_lon as x and fl_lat as y

        # # Set limits for x and y axes
        # plt.xlim(-90, 90)
        # plt.ylim(-90, 90)

        # # Draw a circle centered at (0, 0) with radius 90
        # circle = plt.Circle((0, 0), 90, color='blue', fill=False)
        # plt.gca().add_artist(circle)

        # plt.title('Flare Locations')  # Title of the plot
        # plt.xlabel('Longitude (fl_lon)')  # X-axis label
        # plt.ylabel('Latitude (fl_lat)')  # Y-axis label
        # plt.grid(True)  # Show grid
        # plt.savefig("fl2cme_flare_location_soho.jpg")  # Display the plot

        # Min and max time of flares
        min_time_flares = np.min(df_filtered.start_time)
        max_time_flares = np.max(df_filtered.end_time)

        # Min and max time for the labels (0, 12, 24, 36, 48)
        min_time = min_time_flares.replace(
            minute=(min_time_flares.minute // 12) * 12, second=0, microsecond=0
        )
        max_time = max_time_flares.replace(
            minute=(max_time_flares.minute // 12) * 12 + 12, second=0, microsecond=0
        )

        # List of goes classes
        unique_goes_class = list(df_filtered["goes_class"].unique())

        # List of all timestamps
        timestamps = pd.date_range(start=min_time, end=max_time, freq="12min")

        # Create new DataFrame with 'timestamp' column
        df_flare_timestamp = pd.DataFrame(timestamps, columns=["timestamp"])

        # Create a column for storing class info
        df_flare_timestamp["flare_class"] = np.nan

        #   df_flare_timestamp[(df_filtered["start_time"]>min_time) & (df_filtered["end_time"]<max_time)]
        # assuming start & end date the same
        fl_year = list()
        fl_month = list()
        fl_day = list()
        for i in range(len(df_filtered)):
            fl_year.append(df_filtered["start_time"][i].year)
            fl_month.append(df_filtered["start_time"][i].month)
            fl_day.append(df_filtered["start_time"][i].day)

        # Simplest version for classification
        # Use pandas to merge df_filtered with df_flare_timestamps
        # Find the closest date to peak_time and fill with the class name

        # # Create a dictionary where each key is a GOES class and each value is a Series of NaNs (or 0s) with the same index as df_flare_timestamp
        # columns_to_add = {goes_class: pd.Series(np.nan, index=df_flare_timestamp.index) for goes_class in unique_goes_class}

        # # Use pd.concat to add all the new columns at once
        # df_flare_timestamp = pd.concat([df_flare_timestamp, pd.DataFrame(columns_to_add)], axis=1)

        # # For regression
        # # fill the df with a time to the next flare and give 0 if flare is happening (i.e. it is in between start and end)

        # # For classificiation
        # # Fill the df with a tensor that shows if there is a flare within an hour for next 24 hours

        breakpoint()
