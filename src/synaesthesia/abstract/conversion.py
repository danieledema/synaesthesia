from datetime import datetime


def convert_to_timestamp(timestamp) -> int:
    """Converts a timestamp in the format of 'YYYYMMDDTHHMMSSffffff' to a unix timestamp in seconds."""
    format = "%Y%m%dT%H%M%S%f"
    date = datetime.strptime(timestamp, format)
    return int(date.timestamp())


def convert_to_string(timestamp) -> str:
    """Converts a unix timestamp in seconds to a string in the format of 'YYYYMMDDTHHMMSSfff'."""
    dt_object = datetime.fromtimestamp(timestamp)
    return dt_object.strftime("%Y%m%dT%H%M%S%f")[:-3]
