from datetime import datetime


def convert_to_datetime(timestamp):
    # Adjust this function based on the format of your timestamps
    format = "%Y%m%dT%H%M%S%f"
    return datetime.strptime(timestamp, format)
