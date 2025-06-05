import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import generate_sample_packet, find_packet_start


def test_generate_sample_packet_length():
    sr = 8000
    duration = 0.5
    packet = generate_sample_packet(duration, sr, frequency=1000)
    assert len(packet) == int(sr * duration)


def test_find_packet_start_energy():
    sig = np.concatenate([np.zeros(100), np.ones(50), np.zeros(20)])
    start = find_packet_start(sig)
    assert 98 <= start <= 102


def test_find_packet_start_template():
    template = np.array([1.0, 1.0, 1.0])
    sig = np.concatenate([np.zeros(10), template, np.zeros(5)])
    start = find_packet_start(sig, template=template)
    assert start == 10
