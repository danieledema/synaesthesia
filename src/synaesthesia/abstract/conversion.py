from datetime import datetime
from typing import Union

# Default format constant to avoid repetition and make it easily configurable
DEFAULT_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%f"


def convert_to_timestamp(
    timestamp: Union[str, datetime], format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> int:
    """
    Converts a timestamp string or datetime object to a unix timestamp in seconds.

    Args:
        timestamp: String timestamp or datetime object to convert
        format_str: Format string for parsing (if timestamp is string)

    Returns:
        Unix timestamp as integer seconds

    Raises:
        ValueError: If timestamp string doesn't match the expected format
        TypeError: If timestamp is neither string nor datetime
    """
    if isinstance(timestamp, datetime):
        return int(timestamp.timestamp())
    elif isinstance(timestamp, str):
        try:
            date = datetime.strptime(timestamp, format_str)
            return int(date.timestamp())
        except ValueError as e:
            raise ValueError(
                f"Invalid timestamp format. Expected '{format_str}', got '{timestamp}'"
            ) from e
    else:
        raise TypeError(f"Expected str or datetime, got {type(timestamp)}")


def convert_to_string(
    timestamp: Union[int, float, datetime], format_str: str = DEFAULT_TIMESTAMP_FORMAT
) -> str:
    """
    Converts a unix timestamp or datetime object to a formatted string.

    Args:
        timestamp: Unix timestamp (int/float) or datetime object
        format_str: Output format string

    Returns:
        Formatted timestamp string (truncated to milliseconds if using default format)

    Raises:
        ValueError: If timestamp is invalid
        TypeError: If timestamp is not int, float, or datetime
    """
    if isinstance(timestamp, datetime):
        dt_object = timestamp
    elif isinstance(timestamp, (int, float)):
        try:
            dt_object = datetime.fromtimestamp(timestamp)
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid timestamp value: {timestamp}") from e
    else:
        raise TypeError(f"Expected int, float, or datetime, got {type(timestamp)}")

    formatted = dt_object.strftime(format_str)

    # Truncate microseconds to milliseconds for default format
    if format_str == DEFAULT_TIMESTAMP_FORMAT:
        return formatted[:-3]

    return formatted


def convert_timestamp_format(
    timestamp: str, input_format: str, output_format: str = DEFAULT_TIMESTAMP_FORMAT
) -> str:
    """
    Converts a timestamp string from one format to another.

    Args:
        timestamp: Input timestamp string
        input_format: Format of the input timestamp
        output_format: Desired output format

    Returns:
        Reformatted timestamp string
    """
    dt_object = datetime.strptime(timestamp, input_format)
    formatted = dt_object.strftime(output_format)

    # Apply millisecond truncation for default format
    if output_format == DEFAULT_TIMESTAMP_FORMAT:
        return formatted[:-3]

    return formatted
