import numpy as np
from utils import adjust_packet_bounds_gui, generate_sample_packet

# Create a simple test signal
sample_rate = 56e6  # 56 MHz
duration = 0.001    # 1 ms
signal = generate_sample_packet(duration, sample_rate, 10e6)  # 10 MHz signal

print(f"Signal length: {len(signal)} samples")
print(f"Duration: {len(signal)/sample_rate*1000:.3f} ms")
print("Testing interactive packet bounds adjustment...")

# Test the interactive function
try:
    start_sample, end_sample = adjust_packet_bounds_gui(signal, sample_rate, 
                                                       start_sample=1000, 
                                                       end_sample=len(signal)-1000)
    print(f"Selected range: {start_sample} to {end_sample}")
    print(f"Duration: {(end_sample-start_sample)/sample_rate*1e6:.1f} μs")
except Exception as e:
    print(f"Error: {e}") 