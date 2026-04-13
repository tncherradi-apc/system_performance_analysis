import numpy as np
import matplotlib.pyplot as plt

np.random.seed(None)

# -----------------------------
# 1. Time axis
# -----------------------------
time = np.arange(0, 200, 1)
n = len(time)

# -----------------------------
# 2. Generate a genuinely random baseline
# -----------------------------
base_level = np.random.uniform(45, 55)

# One or two waves mixed together
amp1 = np.random.uniform(1.0, 4.0)
freq1 = np.random.uniform(0.04, 0.14)
phase1 = np.random.uniform(0, 2 * np.pi)

use_second_wave = np.random.rand() < 0.7
if use_second_wave:
    amp2 = np.random.uniform(0.5, 2.5)
    freq2 = np.random.uniform(0.02, 0.09)
    phase2 = np.random.uniform(0, 2 * np.pi)
else:
    amp2 = 0.0
    freq2 = 0.0
    phase2 = 0.0

expected = (
    base_level
    + amp1 * np.sin(freq1 * time + phase1)
    + amp2 * np.sin(freq2 * time + phase2)
)

# Optional slow drift
if np.random.rand() < 0.5:
    drift = np.linspace(
        np.random.uniform(-2.0, 2.0),
        np.random.uniform(-2.0, 2.0),
        n
    )
    expected = expected + drift

# Noise
noise_std = np.random.uniform(0.4, 1.3)
signal = expected + np.random.normal(0, noise_std, size=n)

# -----------------------------
# 3. Inject random faults
# -----------------------------
fault_log = []
used_ranges = []

def overlaps(start, end, existing_ranges, gap=6):
    for s, e in existing_ranges:
        if not (end < s - gap or start > e + gap):
            return True
    return False

num_faults = np.random.randint(2, 7)   # 2 to 6 faults

for _ in range(num_faults):
    fault_type = np.random.choice(
        ["spike", "drop", "rise", "flatline", "noisy_burst"],
        p=[0.30, 0.22, 0.18, 0.15, 0.15]
    )

    placed = False
    for _ in range(100):
        if fault_type == "spike":
            idx = np.random.randint(5, n - 5)
            if overlaps(idx, idx, used_ranges):
                continue

            magnitude = np.random.uniform(5, 12) * np.random.choice([-1, 1])
            signal[idx] += magnitude
            used_ranges.append((idx, idx))
            fault_log.append({
                "type": "spike",
                "start": idx,
                "end": idx,
                "magnitude": magnitude
            })
            placed = True
            break

        elif fault_type in ["drop", "rise"]:
            duration = np.random.randint(5, 25)
            start = np.random.randint(5, n - duration - 5)
            end = start + duration - 1
            if overlaps(start, end, used_ranges):
                continue

            base_mag = np.random.uniform(3, 8)
            magnitude = -base_mag if fault_type == "drop" else base_mag

            # Sometimes abrupt, sometimes ramped
            if np.random.rand() < 0.5:
                signal[start:end + 1] += magnitude
            else:
                ramp = np.linspace(0, magnitude, duration)
                signal[start:end + 1] += ramp

            used_ranges.append((start, end))
            fault_log.append({
                "type": fault_type,
                "start": start,
                "end": end,
                "magnitude": magnitude
            })
            placed = True
            break

        elif fault_type == "flatline":
            duration = np.random.randint(6, 20)
            start = np.random.randint(5, n - duration - 5)
            end = start + duration - 1
            if overlaps(start, end, used_ranges):
                continue

            flat_value = signal[start - 1] + np.random.uniform(-1.0, 1.0)
            signal[start:end + 1] = flat_value + np.random.normal(0, 0.05, size=duration)

            used_ranges.append((start, end))
            fault_log.append({
                "type": "flatline",
                "start": start,
                "end": end,
                "magnitude": flat_value
            })
            placed = True
            break

        elif fault_type == "noisy_burst":
            duration = np.random.randint(6, 20)
            start = np.random.randint(5, n - duration - 5)
            end = start + duration - 1
            if overlaps(start, end, used_ranges):
                continue

            extra_noise = np.random.normal(
                0,
                np.random.uniform(2.0, 5.0),
                size=duration
            )
            signal[start:end + 1] += extra_noise

            used_ranges.append((start, end))
            fault_log.append({
                "type": "noisy_burst",
                "start": start,
                "end": end,
                "magnitude": float(np.std(extra_noise))
            })
            placed = True
            break

# -----------------------------
# 4. Detect faults using expected baseline
# -----------------------------
residual = signal - expected

mad = np.median(np.abs(residual - np.median(residual)))
sigma = 1.4826 * mad
if sigma < 1e-6:
    sigma = np.std(residual)

point_threshold = 4.0 * sigma
region_threshold = 2.5 * sigma

spike_mask = np.abs(residual) > point_threshold
region_mask = np.abs(residual) > region_threshold

# Require persistence for region-type faults
persistent_mask = (
    np.convolve(region_mask.astype(int), np.ones(4, dtype=int), mode='same') >= 3
)

anomaly_mask = spike_mask | persistent_mask
anomaly_indices = np.where(anomaly_mask)[0]

# -----------------------------
# 5. Group detected regions
# -----------------------------
detected_ranges = []
in_region = False
region_start = None

for i, val in enumerate(persistent_mask):
    if val and not in_region:
        region_start = i
        in_region = True
    elif not val and in_region:
        detected_ranges.append((region_start, i - 1))
        in_region = False

if in_region:
    detected_ranges.append((region_start, n - 1))

# -----------------------------
# 6. Plot
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(time, signal, label="Raw Sensor Data", alpha=0.7)
plt.plot(time, expected, label="Expected Baseline", linewidth=2)

# Detected point faults
detected_spikes = np.where(spike_mask)[0]
plt.scatter(
    time[detected_spikes],
    signal[detected_spikes],
    s=80,
    label="Detected Point Faults"
)

# Detected regions
for i, (start, end) in enumerate(detected_ranges):
    if i == 0:
        plt.axvspan(time[start], time[end], alpha=0.18, label="Detected Fault Region")
    else:
        plt.axvspan(time[start], time[end], alpha=0.18)

plt.xlabel("Time")
plt.ylabel("System Output")
plt.title("Randomized System Performance and Fault Detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Print scenario summary
# -----------------------------
print("=== Scenario parameters ===")
print(f"Base level: {base_level:.2f}")
print(f"Wave 1: amplitude={amp1:.2f}, frequency={freq1:.3f}, phase={phase1:.2f}")
if use_second_wave:
    print(f"Wave 2: amplitude={amp2:.2f}, frequency={freq2:.3f}, phase={phase2:.2f}")
else:
    print("Wave 2: not used")
print(f"Noise std: {noise_std:.2f}")

print("\n=== Injected faults ===")
if fault_log:
    for i, f in enumerate(fault_log, start=1):
        print(
            f"{i}. {f['type']} | start={f['start']} | end={f['end']} | magnitude={f['magnitude']:.2f}"
        )
else:
    print("No faults injected")

print("\n=== Detected point-fault indices ===")
print(detected_spikes)

print("\n=== Detected fault regions ===")
if detected_ranges:
    for start, end in detected_ranges:
        print(f"{start} to {end}")
else:
    print("None")
