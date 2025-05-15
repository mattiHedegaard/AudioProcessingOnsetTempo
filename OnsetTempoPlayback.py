import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd

# Load audio file
y, sr = sf.read('disco.00004.wav')
if y.ndim > 1:
    y = y.mean(axis=1)  # convert to mono if stereo

# Parameters
N = 1024       # window size
H = 512        # hop size
threshold_ratio = 0.3  # onset threshold ratio

# Compute energy in overlapping windows
energies = []
positions = []
for m in range(0, len(y) - N, H):
    window = y[m:m + N]
    energy = np.sum(window ** 2)
    energies.append(energy)
    positions.append(m)

energies = np.array(energies)
positions = np.array(positions)

# Compute difference in energy
delta_energy = np.diff(energies, prepend=energies[0])  # ΔE[k] = E[k] - E[k - 1]

# Detect onsets
threshold = np.max(delta_energy) * threshold_ratio
onset_indices = np.where(delta_energy > threshold)[0]
onset_sample_positions = positions[onset_indices]
onset_times = onset_sample_positions / sr

# Estimate tempo
if len(onset_times) > 1:
    intervals = np.diff(onset_times)
    avg_interval = np.mean(intervals)
    tempo_bpm = 60.0 / avg_interval
else:
    tempo_bpm = 0.0

# Output results
print("Detected Onsets (seconds):", onset_times)
print("Estimated Tempo (BPM):", round(tempo_bpm, 2))

# Plot
plt.figure(figsize=(10, 4))
times = positions / sr
plt.plot(times, delta_energy, label='ΔEnergy')
plt.vlines(onset_times, ymin=min(delta_energy), ymax=max(delta_energy), color='r', linestyle='--', label='Onsets')
plt.title('Detected onsets')
plt.xlabel('Time (s)')
plt.legend()
plt.tight_layout()
plt.show(block=False)

# ----------- Add Click Sounds at Onsets -----------

# Create a short beep (e.g., 1 kHz sine wave for 0.05s)
def generate_beep(sr, duration=0.05, freq=1000):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beep = 0.5 * np.sin(2 * np.pi * freq * t)
    return beep

beep = generate_beep(sr)
click_track = np.zeros_like(y)

for pos in onset_sample_positions:
    if pos + len(beep) < len(click_track):
        click_track[pos:pos + len(beep)] += beep

# Mix audio and click (adjust volume as needed)
mixed = y + click_track
mixed = np.clip(mixed, -1.0, 1.0)  # prevent clipping

print("Playing audio with onset clicks")
sd.play(mixed, sr)
sd.wait()
print("Done")

import librosa
import librosa.display

# Convert to float32 for librosa
y_librosa = y.astype(np.float32)

# ----------- MEL SPECTROGRAM -----------

S = librosa.feature.melspectrogram(y=y_librosa, sr=sr, n_fft=N, hop_length=H, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, hop_length=H, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.show(block=False)

# ----------- TEMPOGRAM -----------

# Onset strength envelope
onset_env = librosa.onset.onset_strength(y=y_librosa, sr=sr, hop_length=H)

# Tempogram
tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=H)

plt.figure(figsize=(10, 4))
librosa.display.specshow(tempogram, sr=sr, hop_length=H, x_axis='time', y_axis='tempo')
plt.title('Tempogram')
plt.colorbar(label='Autocorrelation')
plt.tight_layout()
plt.show(block=False)
plt.waitforbuttonpress()