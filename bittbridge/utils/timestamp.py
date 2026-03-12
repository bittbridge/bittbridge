from datetime import datetime, timedelta
from typing import Optional, Union

from pytz import timezone

###############################
#           GETTERS           #
###############################


def get_timezone() -> timezone:
    """
    Global timezone for all timestamp manipulation.
    Eastern (America/New_York) to align with ISO-NE API and New England demand.
    """
    return timezone("America/New_York")


def get_now() -> datetime:
    """
    Get the current datetime
    """
    return datetime.now(get_timezone())


def get_before(
    timestamp: Optional[Union[datetime, str, float]] = None,
    days: int = 0,
    hours: int = 0,
    minutes: int = 5,
    seconds: int = 0,
) -> datetime:
    """
    Get the datetime x minutes before now
    """
    if timestamp is None:
        timestamp = get_now()
    else:
        timestamp = to_datetime(timestamp)

    # Perform the time subtraction
    before_timestamp = timestamp - timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

    return before_timestamp


def get_midnight() -> datetime:
    """
    Get the most recent instance of midnight
    """
    return get_now().replace(hour=0, minute=0, second=0, microsecond=0)


def get_posix() -> float:
    """
    Get the current POSIX time, seconds that have elapsed since Jan 1 1970
    """
    return to_posix(get_now())


def get_str() -> str:
    """
    Get the current timestamp as a string, convenient for requests
    """
    return to_str(get_now())


###############################
#         CONVERTERS          #
###############################


def to_posix(timestamp: Union[datetime, str, float]) -> float:
    """
    Convert datetime to seconds that have elapsed since Jan 1 1970
    """

    dt = to_datetime(timestamp)
    posix_timestamp = dt.timestamp()

    return float(posix_timestamp)


def to_str(timestamp: Union[datetime, str, float]) -> str:
    """
    Convert datetime to ISO 8601 string (with timezone offset).
    """
    dt = to_datetime(timestamp)
    return dt.isoformat()


def to_datetime(timestamp: Union[str, float]) -> datetime:
    """
    Convert ISO 8601 string (with Z or offset) or POSIX float to datetime in global timezone (Eastern).
    """
    if isinstance(timestamp, str):
        # Accept both "Z" and offset formats (e.g. -04:00)
        ts = timestamp.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone("UTC"))
        return dt.astimezone(get_timezone())

    if isinstance(timestamp, float):
        return datetime.fromtimestamp(timestamp, tz=get_timezone())

    if isinstance(timestamp, datetime):
        return timestamp.astimezone(get_timezone())

    raise TypeError(
        "Must pass a timestamp that is either a iso 8601 string, POSIX time float, or a datetime object"
    )


###############################
#          FUNCTIONS          #
###############################


def elapsed_seconds(timestamp1: datetime, timestamp2: datetime) -> float:
    """
    Absolute number of seconds between two timestamps
    """
    return abs((timestamp1 - timestamp2).total_seconds())


def round_minute_down(timestamp: datetime, base: int = 5) -> datetime:
    """
    Round the timestamp down to the nearest 5 minutes

    Example:
        >>> timestamp = datetime.now(timezone("America/New_York"))
        >>> round_minute_down(timestamp)
        datetime with minute rounded down to multiple of base
    """
    # Round the minute down to the nearest multiple of `base`
    correct_minute: int = timestamp.minute // base * base
    return timestamp.replace(minute=correct_minute, second=0, microsecond=0)


def get_next_interval(interval_minutes: int = 5) -> datetime:
    """
    Get the start of the next 5-minute slot (Eastern).
    Use when the validator asks "what will demand be in the NEXT 5-min slot?".
    E.g. at 10:00:00 returns 10:05:00; at 10:04:59 returns 10:05:00; at 10:05:00 returns 10:10:00.
    """
    now = get_now()
    rounded_down = round_minute_down(now, base=interval_minutes)
    return rounded_down + timedelta(minutes=interval_minutes)


def round_to_interval(timestamp: datetime, interval_minutes: int = 5) -> datetime:
    """
    Round a timestamp to the nearest interval (up or down).

    Args:
        timestamp (datetime): The timestamp to round
        interval_minutes (int): The interval in minutes to round to. Defaults to 5.

    Returns:
        datetime: A new timestamp rounded to the nearest interval

    Example:
        >>> dt in Eastern; round_to_interval(dt, 5)  # Rounds to nearest 5 min in Eastern
    """
    if not isinstance(timestamp, datetime):
        timestamp = to_datetime(timestamp)

    # Work in global timezone (Eastern)
    local_timestamp = timestamp.astimezone(get_timezone())

    # Calculate total minutes since midnight (Eastern)
    minutes_since_midnight = local_timestamp.hour * 60 + local_timestamp.minute

    # Calculate the nearest interval
    rounded_minutes = round(minutes_since_midnight / interval_minutes) * interval_minutes

    # Create new timestamp with rounded minutes
    new_timestamp = local_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        minutes=rounded_minutes
    )

    return new_timestamp


def is_query_time(prediction_interval: int, timestamp: str, tolerance: int = 120) -> bool:
    """
    Tolerance - in seconds, how long to allow after epoch start
    prediction_interval - in minutes, how often to predict

    Notes:
        This function will be called multiple times
        First, check that we are in a new epoch
        Then, check if we already sent a request in the current epoch
    """
    now: datetime = get_now()
    provided_timestamp: datetime = to_datetime(timestamp)

    # The provided timestamp is the last time a request was made. If this timestamp
    # is from the current epoch, we do not want to make a request. One way to check
    # this is by checking that `now` and `provided_timestamp` are more than `tolerance`
    # apart from each other. When true, this means the `provided_timestamp` is from
    # the previous epoch
    been_long_enough = elapsed_seconds(now, provided_timestamp) > tolerance

    # return false early if we already know it has not been long enough
    if not been_long_enough:
        return False

    # If it has been long enough, let's check the epoch start time
    midnight = get_midnight()
    sec_since_open = elapsed_seconds(now, midnight)

    # To check if this is a new epoch, compare the current timestamp
    # to the expected beginning of an epoch. If we are within `tolerance`
    # seconds of a new epoch, then we are willing to send a request
    sec_since_epoch_start = sec_since_open % (prediction_interval * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance

    # We know a request hasn't been sent yet, so simply return T/F based
    # on beginning of epoch
    return beginning_of_epoch
