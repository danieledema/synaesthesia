import pandas as pd
import matplotlib.pyplot as plt


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

    df = pd.read_csv(path)

    print(df.columns)

    # Index(['flare_id', 'start_time', 'peak_time', 'end_time', 'goes_class',
    #    'noaa_active_region', 'fl_lon', 'fl_lat', 'fl_loc_src', 'ssw_flare_id',
    #    'hinode_flare_id', 'primary_verified', 'secondary_verified',
    #    'candidate_ars', 'cme_id', 'fl_pa', 'cme_mpa', 'diff_a', 'cme_vel',
    #    'cme_width', 'cme_assoc_conf', 'cdaw_cme_id', 'cdaw_cme_width',
    #    'cdaw_cme_vel', 'cdaw_cme_pa', 'donki_cme_id', 'donki_cme_half_angle',
    #    'donki_cme_vel', 'lowcat_cme_id', 'lowcat_cme_width', 'lowcat_cme_vel',
    #    'lowcat_cme_pa', 'cme_valid_conf'],
    #   dtype='object')
