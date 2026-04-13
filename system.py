import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate simulated system data
# -----------------------------
np.random.seed(42)

time = np.arange(0, 200, 1)

# Baseline system behavior: small oscillation around a normal value
baseline = 50 + 2 * np.sin(0.1 * time)

# Add random measurement noise
noise = np.random.normal(0, 0.8, size=len(time))

signal = baseline + noise

# Inject anomalies / faults
signal[50] += 8         # sudden spike
signal[120:130] -= 5    # temporary performance drop
signal[170] += 10       # major spike

# -----------------------------
# 2. Smooth the signal
# -----------------------------
window_size = 5
smoothed = np.convolve(signal, np.ones(window_size) / window_size, mode='same')

# -----------------------------
# 3. Detect anomalies
# -----------------------------
# Compare original signal to smoothed trend
residual = signal - smoothed

# Set anomaly threshold using standard deviation
threshold = 2.5 * np.std(residual)

anomaly_indices = np.where(np.abs(residual) > threshold)[0]

# -----------------------------
# 4. Plot results
# -----------------------------
plt.figure(figsize=(12, 7))

# Original and smoothed signal
plt.plot(time, signal, label="Raw Sensor Data", alpha=0.7)
plt.plot(time, smoothed, label="Smoothed Trend", linewidth=2)

# Highlight anomalies
plt.scatter(
    time[anomaly_indices],
    signal[anomaly_indices],
    label="Detected Faults",
    marker='o',
    s=80
)

plt.xlabel("Time")
plt.ylabel("System Output")
plt.title("System Performance Analysis and Fault Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Print anomaly summary
# -----------------------------
print("Detected anomaly indices:")
print(anomaly_indices)

print("\nDetected anomaly values:")
print(signal[anomaly_indices])
