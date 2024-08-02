from datetime import datetime


def convert_to_timestamp(timestamp, format="%Y%m%dT%H%M%S%f") -> int:
    """Converts a timestamp to a unix timestamp in seconds."""
    date = datetime.strptime(timestamp, format)
    return int(date.timestamp())


def convert_to_string(timestamp) -> str:
    """Converts a unix timestamp in seconds to a string in the format of 'YYYYMMDDTHHMMSSfff'."""
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object.strftime("%Y%m%dT%H%M%S%f")[:-3]
