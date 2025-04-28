from datetime import timedelta

from snp_oracle.utils.timestamp import get_now, get_timezone, round_minute_down, to_datetime, to_str


class MinerHistory:
    """This class is used to store miner predictions along with their timestamps.
    Allows for easy formatting, filtering, and lookup of predictions by timestamp.
    """

    def __init__(self, uid: int, timezone=get_timezone()):
        self.predictions = {}
        self.directions = {}
        self.uid = uid
        self.timezone = timezone

    def add_prediction(self, timestamp, prediction: float, direction: bool):
        if isinstance(timestamp, str):
            timestamp = to_datetime(timestamp)
        timestamp = round_minute_down(timestamp)
        if prediction is not None:
            self.predictions[to_str(timestamp)] = float(prediction)
        if direction is not None:
            self.directions[to_str(timestamp)] = bool(direction)

    def clear_old_predictions(self):
        # deletes predictions older than 24 hours
        start_time = round_minute_down(get_now()) - timedelta(hours=24)
        self.predictions = {key: value for key, value in self.predictions.items() if start_time <= to_datetime(key)}
        self.directions = {key: value for key, value in self.directions.items() if start_time <= to_datetime(key)}

    def get_prediction(self, reference_timestamp):
        ts = to_str(reference_timestamp)
        return self.predictions.get(ts, None), self.directions.get(ts, None)
