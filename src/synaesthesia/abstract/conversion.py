from datetime import datetime


def convert_to_timestamp(timestamp) -> int:
    """Converts a timestamp in the format of 'YYYYMMDDTHHMMSSffffff' to a unix timestamp in seconds."""
    format = "%Y%m%dT%H%M%S%f"
    date = datetime.strptime(timestamp, format)
    return int(date.timestamp())
