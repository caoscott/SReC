import subprocess
import time
from contextlib import contextmanager
from typing import List


class TimeAccumulator(object):
    """
    Usage:
    t = TimeAccumulator()
    for i, x in enumerate(it):
        with t.execute():
            expensive_operation()
        if i % 50 == 0:
            print('Average: {}'.format(t.mean_time_spent()))
    """
    _EPS = 1e-8

    def __init__(self) -> None:
        self.times: List[float] = []

    @contextmanager
    def execute(self):
        prev = time.time()
        try:
            yield
            self.times.append(time.time() - prev)
        except subprocess.CalledProcessError:
            raise

    def mean_time_spent(self) -> float:
        """ :returns mean time spent and resets cached times. """
        total_time_spent = sum(self.times)
        count = float(len(self.times))
        if count == 0:
            count += self._EPS
        # self.times = []
        # prevent div by zero errors
        return total_time_spent / count
