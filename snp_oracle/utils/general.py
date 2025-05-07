import asyncio
import time
from typing import Any, Callable

import bittensor as bt
from numpy import argsort, array, concatenate, cumsum, empty_like


async def loop_handler(self, func: Callable, sleep_time: float = 120):
    try:
        while not self.stop_event.is_set():
            async with self.lock:
                await func()
            await asyncio.sleep(sleep_time)
    except asyncio.CancelledError:
        bt.logging.error(f"{func.__name__} cancelled")
        raise
    except KeyboardInterrupt:
        raise
    except Exception as e:
        bt.logging.error(f"{func.__name__} raised error: {e}")
        raise e
    finally:
        async with self.lock:
            self.stop_event.set()


def func_with_retry(func: Callable, max_attempts: int = 3, delay: float = 1, *args, **kwargs) -> Any:
    attempt = 0
    while attempt < max_attempts:
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            attempt += 1
            bt.logging.debug(f"Function {func} failed: Attempt {attempt} of {max_attempts} with error: {e}")
            if attempt == max_attempts:
                bt.logging.error(f"Function {func} failed {max_attempts} times, skipping.")
                raise
            else:
                time.sleep(delay)


def rank(vector):
    if vector is None or len(vector) <= 1:
        return array([0])
    else:
        # Sort the array and get the indices that would sort it
        sorted_indices = argsort(vector)
        sorted_vector = vector[sorted_indices]
        # Create a mask for where each new unique value starts in the sorted array
        unique_mask = concatenate(([True], sorted_vector[1:] != sorted_vector[:-1]))
        # Use cumulative sum of the unique mask to get the ranks, then assign back in original order
        ranks = cumsum(unique_mask) - 1
        rank_vector = empty_like(vector, dtype=int)
        rank_vector[sorted_indices] = ranks
        return rank_vector
