"""LEAP and passive knob transport adapted from ARIA deployment_scripts/leap_hand_hw.py.

Only this module opens Dynamixel ports; callers must explicitly enable hardware.
"""

import glob
import math
import time

import torch

from src.policy.knob_interface import JOINT_LOWER, JOINT_UPPER, REAL_TO_SIM, SIM_TO_REAL

BAUDRATE = 4_000_000
POSITION_SCALE = 2 * math.pi / 4096


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class MotorBus:
    def __init__(self, port, motor_ids):
        import dynamixel_sdk as dxl

        self.dxl = dxl
        self.port = dxl.PortHandler(port)
        self.packet = dxl.PacketHandler(2.0)
        self.ids = list(motor_ids)
        self.reader = dxl.GroupBulkRead(self.port, self.packet)
        for motor in self.ids:
            if not self.reader.addParam(motor, 132, 4):
                raise RuntimeError(f"Cannot configure position reader for motor {motor}")

    def connect(self):
        if not self.port.openPort() or not self.port.setBaudRate(BAUDRATE):
            self.port.closePort()
            raise OSError("Cannot open Dynamixel port at 4 Mbps; check device permissions and connection")

    def write_register(self, motor, address, value, size):
        write = getattr(self.packet, f"write{size}ByteTxRx")
        for _ in range(3):
            result, error = write(self.port, motor, address, int(value))
            if result == self.dxl.COMM_SUCCESS and error == 0:
                return
        raise OSError(f"Dynamixel write failed: motor={motor}, address={address}, result={result}, error={error}")

    def read_positions(self):
        for _ in range(2):
            result = self.reader.txRxPacket()
            if result == self.dxl.COMM_SUCCESS and all(self.reader.isAvailable(m, 132, 4) for m in self.ids):
                values = [self.reader.getData(m, 132, 4) for m in self.ids]
                return torch.tensor([v - 2**32 if v >= 2**31 else v for v in values], dtype=torch.float32) * POSITION_SCALE
        raise OSError("Dynamixel position read failed or returned incomplete motor data")

    def command(self, positions):
        writer = self.dxl.GroupSyncWrite(self.port, self.packet, 116, 4)
        try:
            for motor, position in zip(self.ids, positions, strict=True):
                value = round(float(position) / POSITION_SCALE)
                if not writer.addParam(motor, list(value.to_bytes(4, "little", signed=True))):
                    raise OSError(f"Cannot queue position for motor {motor}")
            if writer.txPacket() != self.dxl.COMM_SUCCESS:
                raise OSError("Dynamixel target write failed")
        finally:
            writer.clearParam()

    def disconnect(self):
        self.port.closePort()


def detect_ports(hand_port=None, knob_port=None):
    if hand_port and knob_port:
        if hand_port == knob_port:
            raise ValueError("Hand and knob require separate ports")
        return hand_port, knob_port
    import dynamixel_sdk as dxl

    for path in sorted(glob.glob("/dev/ttyUSB*")):
        if path in (hand_port, knob_port):
            continue
        port, packet = dxl.PortHandler(path), dxl.PacketHandler(2.0)
        try:
            if not port.openPort() or not port.setBaudRate(BAUDRATE):
                continue
            def responds(motor):
                _, result, error = packet.ping(port, motor)
                return result == dxl.COMM_SUCCESS and error == 0
            if knob_port is None and responds(50):
                knob_port = path
            elif hand_port is None and all(responds(m) for m in (0, 8)):
                hand_port = path
        finally:
            port.closePort()
        if hand_port and knob_port:
            break
    if not hand_port or not knob_port:
        raise RuntimeError("Cannot detect both devices; supply --hand-port and --knob-port")
    return hand_port, knob_port


class LeapHand:
    def __init__(self, port, *, joint_offset=math.pi, kp=800, ki=0, kd=200, current_limit=500):
        self.bus = MotorBus(port, range(16))
        self.offset = joint_offset
        self.gains = (kp, ki, kd, current_limit)
        self.connected = False

    def connect(self):
        self.bus.connect()
        self.connected = True
        kp, ki, kd, current = self.gains
        # Configure with torque off, seed the measured pose, then enable torque.
        for motor in range(16):
            self.bus.write_register(motor, 64, 0, 1)
            self.bus.write_register(motor, 11, 5, 1)
            for address, value in ((84, kp * (0.75 if motor in (0, 4, 8) else 1)),
                                   (82, ki), (80, kd * (0.75 if motor in (0, 4, 8) else 1)), (102, current)):
                self.bus.write_register(motor, address, value, 2)
        self.bus.command(self.bus.read_positions())
        for motor in range(16):
            self.bus.write_register(motor, 64, 1, 1)

    def poll_joint_position(self):
        return (self.bus.read_positions() - self.offset)[REAL_TO_SIM]

    def command_joint_position(self, target):
        target = torch.as_tensor(target).flatten()
        if target.shape != (16,) or not torch.isfinite(target).all():
            raise ValueError("Hand target must contain 16 finite joint positions")
        if torch.any(target < torch.tensor(JOINT_LOWER)) or torch.any(target > torch.tensor(JOINT_UPPER)):
            raise ValueError("Hand target exceeds joint limits")
        self.bus.command(target[SIM_TO_REAL] + self.offset)

    def disconnect(self):
        errors = []
        try:
            if self.connected:
                for motor in range(16):
                    try:
                        self.bus.write_register(motor, 64, 0, 1)
                    except OSError as exc:
                        errors.append(str(exc))
        finally:
            self.connected = False
            self.bus.disconnect()
        if errors:
            raise OSError("; ".join(errors))


class DynamixelKnob:
    def __init__(self, port, *, zero_offset=0.0, direction=1):
        self.bus = MotorBus(port, [50])
        self.zero_offset, self.direction = zero_offset, direction
        self.previous_angle = self.previous_time = None

    def connect(self):
        self.bus.connect()

    def read_position(self):
        angle = wrap(self.direction * (float(self.bus.read_positions()[0]) - self.zero_offset))
        now = time.monotonic()
        velocity = 0.0 if self.previous_angle is None else wrap(angle - self.previous_angle) / max(now - self.previous_time, 1e-6)
        self.previous_angle, self.previous_time = angle, now
        return angle, velocity

    def disconnect(self):
        self.bus.disconnect()


class MockHand:
    def __init__(self):
        self.position = torch.zeros(16)

    def connect(self):
        pass

    def poll_joint_position(self):
        return self.position.clone()

    def command_joint_position(self, target):
        self.position = target.clone()

    def disconnect(self):
        pass


class MockKnob:
    angle = 0.0

    def connect(self):
        pass

    def read_position(self):
        return self.angle, 0.0

    def disconnect(self):
        pass
