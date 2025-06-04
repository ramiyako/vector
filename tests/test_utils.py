import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_sample_packet


def test_generate_sample_packet_length():
    sr = 8000
    duration = 0.5
    packet = generate_sample_packet(duration, sr, frequency=1000)
    assert len(packet) == int(sr * duration)
