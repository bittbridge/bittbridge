from datetime import datetime, timedelta
from typing import Optional, Union

import pandas_market_calendars as mcal
from pytz import timezone

from snp_oracle.constants import PREDICTION_FUTURE_HOURS, PREDICTION_INTERVAL_MINUTES

###############################
#           GETTERS           #
###############################


def get_timezone() -> timezone:
    """
    Set the Global shared timezone for all timestamp manipulation
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

    # Verify datetime object and convert to UTC
    utc_datetime = to_datetime(timestamp)

    # Convert to posix time float
    posix_timestamp = utc_datetime.timestamp()

    return float(posix_timestamp)


def to_str(timestamp: Union[datetime, str, float]) -> str:
    """
    Convert datetime to iso 8601 string
    """

    # Verify datetime object and convert to UTC
    utc_datetime = to_datetime(timestamp)

    # Convert to iso8601 string
    str_datetime = utc_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    return str(str_datetime)


def to_datetime(timestamp: Union[str, float]) -> datetime:
    """
    Convert iso 8601 string, or a POSIX time float, to datetime
    """
    if isinstance(timestamp, str):
        # Assume the proper iso 8601 string format is used
        # `strptime` will trigger an error as needed
        return datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=get_timezone())

    elif isinstance(timestamp, float):
        # Assume proper float value
        # `fromtimestamp` will trigger errors as needed
        return datetime.fromtimestamp(timestamp, tz=get_timezone())

    elif isinstance(timestamp, datetime):
        # Already a datetime object
        # Return as UTC
        return timestamp.astimezone(get_timezone())

    else:
        # Invalid typing
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
        >>> timestamp = datetime.now(timezone("UTC"))
        >>> timestamp
        datetime.datetime(2024, 11, 14, 18, 18, 16, 719482, tzinfo=<UTC>)
        >>> round_minute_down(timestamp)
        datetime.datetime(2024, 11, 14, 18, 15, tzinfo=<UTC>)
    """
    # Round the minute down to the nearest multiple of `base`
    correct_minute: int = timestamp.minute // base * base
    return timestamp.replace(minute=correct_minute, second=0, microsecond=0)


def round_to_interval(timestamp: datetime, interval_minutes: int = 5) -> datetime:
    """
    Round a timestamp to the nearest interval (up or down).

    Args:
        timestamp (datetime): The timestamp to round
        interval_minutes (int): The interval in minutes to round to. Defaults to 5.

    Returns:
        datetime: A new timestamp rounded to the nearest interval

    Example:
        >>> dt = datetime(2024, 1, 1, 14, 13, 30, tzinfo=timezone('UTC'))
        >>> round_to_interval(dt, 5)
        datetime.datetime(2024, 1, 1, 14, 15, tzinfo=<UTC>)  # Rounds up to 14:15
        >>> round_to_interval(dt, 15)
        datetime.datetime(2024, 1, 1, 14, 15, tzinfo=<UTC>)  # Rounds up to 14:15
        >>> dt = datetime(2024, 1, 1, 14, 16, 30, tzinfo=timezone('UTC'))
        >>> round_to_interval(dt, 15)
        datetime.datetime(2024, 1, 1, 14, 15, tzinfo=<UTC>)  # Rounds down to 14:15
    """
    if not isinstance(timestamp, datetime):
        timestamp = to_datetime(timestamp)

    # Ensure timestamp is in UTC
    utc_timestamp = timestamp.astimezone(get_timezone())

    # Calculate total minutes since midnight
    minutes_since_midnight = utc_timestamp.hour * 60 + utc_timestamp.minute

    # Calculate the nearest interval
    rounded_minutes = round(minutes_since_midnight / interval_minutes) * interval_minutes

    # Create new timestamp with rounded minutes
    new_timestamp = utc_timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
        minutes=rounded_minutes
    )

    return new_timestamp


def is_query_time(timestamp: str, tolerance: int = 120) -> bool:
    """
    Tolerance - in seconds, how long to allow after epoch start

    Notes:
        This function will be called multiple times
        First, check that we are in a new epoch
        Then, check if we already sent a request in the current epoch
    """
    now: datetime = get_now()
    provided_timestamp: datetime = to_datetime(timestamp)

    # If the stock market is closed, return False
    if not is_valid_time(now):
        return False

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
    start_of_epoch = now.replace(hour=9, minute=30, second=0, microsecond=0)
    sec_since_open = elapsed_seconds(now, start_of_epoch)

    # To check if this is a new epoch, compare the current timestamp
    # to the expected beginning of an epoch. If we are within `tolerance`
    # seconds of a new epoch, then we are willing to send a request
    sec_since_epoch_start = sec_since_open % (PREDICTION_INTERVAL_MINUTES * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance

    # We are too far away from an epoch to send a request
    if not beginning_of_epoch:
        return False

    end_time = now.replace(hour=16 - PREDICTION_FUTURE_HOURS, minute=2, second=0, microsecond=0)
    if now > end_time:
        return False

    # We know a request hasn't been sent yet, so simply return True
    return True


def is_scoring_time(timestamp: str, tolerance: int = 120):
    """
    Tolerance - in seconds, how long to allow after epoch start

    Notes:
        This function will be called multiple times
        First, check that we are in a new epoch
        Then, check if we already sent a request in the current epoch
    """
    now: datetime = get_now()
    provided_timestamp: datetime = to_datetime(timestamp)

    # If the stock market is closed, return False
    if not is_valid_time(now):
        return False

    # The provided timestamp is the last time a score update was made. If this timestamp
    # is from the current epoch, we do not want to make a score update. One way to check
    # this is by checking that `now` and `provided_timestamp` are more than `tolerance`
    # apart from each other. When true, this means the `provided_timestamp` is from
    # the previous epoch
    been_long_enough = elapsed_seconds(now, provided_timestamp) > tolerance

    # return false early if we already know it has not been long enough
    if not been_long_enough:
        return False

    # If it has been long enough, let's check the epoch start time
    start_of_epoch = now.replace(hour=9, minute=30, second=0, microsecond=0)
    sec_since_open = elapsed_seconds(now, start_of_epoch)

    # To check if this is a new epoch, compare the current timestamp
    # to the expected beginning of an epoch. If we are within `tolerance`
    # seconds of a new epoch, then we are willing to do a score update
    sec_since_epoch_start = sec_since_open % (PREDICTION_INTERVAL_MINUTES * 60)
    beginning_of_epoch = sec_since_epoch_start < tolerance

    # We are too far away from an epoch to send a request
    if not beginning_of_epoch:
        return False

    # Don't update scores in the first hour of market opening
    start_time = now.replace(hour=9 + PREDICTION_FUTURE_HOURS, minute=30, second=0, microsecond=0)
    if now < start_time:
        return False

    # We need to do a score update for this epoch
    return True


def is_valid_time(now) -> bool:
    """
    This function checks if the NYSE is open and validators should send requests.
    The final valid time is 4:00 PM - prediction length (self.INTERVAL) so that the final prediction is for 4:00 PM

    Returns:
        True if the NYSE is open and the current time is between 9:30 AM and (4:00 PM - self.INTERVAL)
        False otherwise

    Notes:
    ------
    Timezone is set to America/New_York

    """
    # Check if today is Monday through Friday
    if now.weekday() >= 5:  # 0 is Monday, 6 is Sunday
        return False
    # Check if the NYSE is open (i.e. not a holiday)
    if not market_is_open(now):
        return False
    # Check if the current time is between 9:30 AM and 4:00 PM
    start_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    end_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if not (start_time <= now <= end_time):
        return False
    # if all checks pass, return true
    return True


def market_is_open(date) -> bool:
    """
    This is an extra check for holidays where the NYSE is closed

    Args:
        date (datetime): The date to check

    Returns:
        True if the NYSE is open.
        False otherwise

    """
    result = mcal.get_calendar("NYSE").schedule(start_date=date, end_date=date)
    return not result.empty
