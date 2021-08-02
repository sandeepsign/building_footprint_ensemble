import time
from common_utils.basic import roundf

# A utility class for keeping track of the elapsed times of various tasks
class TimeKeeper:
    def __init__(self, default_precision=None):
        self.timers = {}
        self.default_precision = default_precision

    # Store the starting time of a task via its name
    def start_timer(self, name):
        if name not in self.timers:
            self.timers[name] = time.time()

    # Check the elapsed time from the start to the end of the task
    def check_timer(self, name):
        if name in self.timers:
            elapsed_time = time.time() - self.timers[name]
            if self.default_precision is None:
                return elapsed_time
            else:
                return roundf(elapsed_time, self.default_precision)
        else:
            return 0

    # Clear an existing timer of a task
    def reset_timer(self, name):
        if name in self.timers:
            del self.timers[name]

    # Clear all existing timer for all tasks
    def reset_all_timers(self):
        self.timers = {}

    # Checks the timer and resets it immediately after
    def check_timer_once(self, name):
        elapsed_time = self.check_timer(name)
        self.reset_timer(name)
        return elapsed_time