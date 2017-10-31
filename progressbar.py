import sys

class ProgressBar(object):

    def __init__(self, max_value, char='-', width=100):
        self.max_value = max_value
        self.value = 0
        self.char = char
        self.width = width
        self._started = False
        self._previous_progress = 0
        self._const = float(self.width) / self.max_value

    def update(self, n=1):
        if not self._started:
            sys.stdout.write('[%s]' % (' ' * self.width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.width + 1))
            self._started = True
        self.value += n
        if self.value > self.max_value:
            self.value = self.max_value
        progress = int(self.value * self._const)
        if progress > self._previous_progress:
            num_steps = progress - self._previous_progress
            self._previous_progress = progress
            sys.stdout.write(self.char * num_steps)
            sys.stdout.flush()
        if self.value == self.max_value:
            sys.stdout.write('\n')

    def reset(self):
        self.value = 0
        self._previous_progress = 0
        self._started = False

    def set_value(self, n):
        self.update(n - self.value)