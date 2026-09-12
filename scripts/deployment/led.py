"""ARIA LED ring protocol; serial acknowledgement never blocks hand control."""

import math
import threading


class LedRing:
    def __init__(self, port, size=88, offset=0, direction=1):
        import serial

        self.serial = serial.Serial(port, 115200, timeout=0.1, write_timeout=0.1)
        self.size, self.offset, self.direction = size, offset, direction
        self.condition = threading.Condition()
        self.pending = {i: (0, 0, 0) for i in range(300)}
        self.previous = set()
        self.stopping = False
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def index(self, angle):
        return (self.offset + self.direction * int((angle + math.pi) / (2 * math.pi) * self.size)) % self.size

    @staticmethod
    def packet(index, color):
        return bytes([index >> 8, index & 255, *color])

    def update(self, angle, target, setup=False):
        if self.error is not None:
            raise OSError("LED serial worker failed") from self.error
        error = abs((angle - target + math.pi) % (2 * math.pi) - math.pi) / math.pi
        colors = {}
        for center, color in ((self.index(target), (255, 255, 255) if setup else (0, 0, 255)),
                              (self.index(angle), (int(255 * error), int(255 * (1 - error)), 0))):
            for delta, weight in zip(range(-2, 3), (0.05, 0.3, 1.0, 0.3, 0.05)):
                index = (center + delta) % self.size
                old = colors.get(index, (0, 0, 0))
                colors[index] = tuple(min(255, a + int(b * weight)) for a, b in zip(old, color))
        with self.condition:
            self.pending.update({i: (0, 0, 0) for i in self.previous - colors.keys()})
            self.pending.update(colors)
            self.previous = set(colors)
            self.condition.notify()

    def _write(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending or self.stopping)
                    if self.stopping:
                        return
                    index = next(iter(self.pending))
                    color = self.pending.pop(index)
                self.serial.write(self.packet(index, color))
                if len(self.serial.read(3)) != 3:
                    raise OSError("LED acknowledgement timed out")
        except Exception as exc:
            self.error = exc

    def disconnect(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join(timeout=0.5)
        self.serial.close()
        if self.thread.is_alive():
            raise RuntimeError("LED worker did not stop")
