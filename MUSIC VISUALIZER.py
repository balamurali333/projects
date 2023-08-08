import numpy as np
import pygame # audio playback library
from pydub import AudioSegment
from scipy.signal import butter, lfilter
import threading
import os
import tkinter as tk
from tkinter import filedialog

# Audio settings
CHUNK_SIZE = 1024 #The number of audio samples processed at a time.
SAMPLE_RATE = 44100 #number of samples per second

# Visualization settings
WINDOW_SIZE = 1024
WINDOW_HEIGHT = 300
BAR_WIDTH = 10
BAR_SPACING = 2
PROGRESS_BAR_HEIGHT = 10

# Low-pass filter settings
CUTOFF_FREQUENCY = 4000  # for noise removal

# Function to apply a low-pass filter to the audio data
def apply_low_pass_filter(data, cutoff_frequency, sample_rate):
    nyquist_frequency = 0.5 * sample_rate
    normal_cutoff = cutoff_frequency / nyquist_frequency
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data.astype(np.int16)

# Function to handle audio playback in a separate thread
def play_audio():
    pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=1)
    pygame.mixer.music.load(temp_wav_file)
    pygame.mixer.music.play(-1)  # -1 means loop indefinitely
    global is_audio_playing
    is_audio_playing = True

# Global variables
is_running = False
start = 0
pcm_data = None
audio_duration = 0
is_paused = False
is_audio_playing = False
color_palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
color_index = 0


# Function to draw the duration bar
def draw_duration_bar(): # progress bar indicating the audio playback progress.
    progress = (pygame.mixer.music.get_pos() / audio_duration) * WINDOW_SIZE
    canvas.coords(progress_bar, 0, WINDOW_HEIGHT - PROGRESS_BAR_HEIGHT, progress, WINDOW_HEIGHT)


# Main visualization loop
def visualize_audio(): # runs in the main thread and continuously updates the visualization.
    global start, color_index, is_paused, is_audio_playing

    while is_running:
        if not is_paused:
            # Read audio data from the stream
            chunk = pcm_data[start: start + CHUNK_SIZE].tobytes()
            samples = np.frombuffer(chunk, dtype=np.int16)

            # Compute the Fast Fourier Transform (FFT)
            fft_result = np.fft.fft(samples) / CHUNK_SIZE
            fft_magnitude = np.abs(fft_result)[:WINDOW_SIZE // 2]

            # Normalize the FFT data
            max_magnitude = np.max(fft_magnitude)
            if max_magnitude != 0:
                fft_magnitude /= max_magnitude
            else:
                fft_magnitude = np.zeros_like(fft_magnitude)

            # Set NaN values to zero
            fft_magnitude = np.nan_to_num(fft_magnitude)

            # Clear the canvas
            canvas.delete("all")

            # Draw the visualization
            for i, magnitude in enumerate(fft_magnitude):
                bar_height = int(magnitude * WINDOW_HEIGHT)
                color = "#{:02x}{:02x}{:02x}".format(*color_palette[color_index % len(color_palette)])
                canvas.create_rectangle(i * (BAR_WIDTH + BAR_SPACING), WINDOW_HEIGHT - bar_height,
                                        i * (BAR_WIDTH + BAR_SPACING) + BAR_WIDTH, WINDOW_HEIGHT, fill=color)

            # Draw the duration bar
            draw_duration_bar()
            # Wait for the next frame
            root.update()
            pygame.time.wait(int(1000 * CHUNK_SIZE / SAMPLE_RATE))

            start += CHUNK_SIZE
            if start >= len(pcm_data):
                stop_visualizer()

            # Increment the color index to change the color for the next frame
            color_index += 1

# Function to start the visualizer
def start_visualizer(): # reading the audio file, applying the filter, and initiating audio playback and visualization.
#Stopping the Visualizer and Audio Playback:
    global is_running, start, color_index, audio_thread, pcm_data, audio_duration, is_paused, is_audio_playing

    mp3_file_path = filedialog.askopenfilename(filetypes=[("MP3 files", "*.mp3")])
    if not mp3_file_path:
        return

    # Check if the visualizer is already running
    if is_running:
        stop_visualizer()
        audio_thread.join()  # Wait for the current audio thread to finish

    # Load the MP3 file using pydub
    audio_data = AudioSegment.from_file(mp3_file_path)

    # Convert the audio data to raw PCM data (16-bit signed integers)
    pcm_data = np.array(audio_data.get_array_of_samples(), dtype=np.int16)

    # Apply low-pass filter to remove background noise
    pcm_data = apply_low_pass_filter(pcm_data, CUTOFF_FREQUENCY, SAMPLE_RATE)

    # Save PCM data to a temporary WAV file
    global temp_wav_file
    temp_wav_file = "temp.wav"
    audio_data.export(temp_wav_file, format="wav")

    # Get the duration of the audio in milliseconds
    audio_duration = len(audio_data)

    # Reset the visualization variables
    start = 0
    color_index = 0
    is_running = True
    is_paused = False
    is_audio_playing = True

    # Start the audio playback in a separate thread
    audio_thread = threading.Thread(target=play_audio)
    audio_thread.start()

    # Start the visualization loop in the main thread
    visualize_audio()

# Function to stop the visualizer and audio playback
def stop_visualizer():
    global is_running, is_audio_playing
    is_running = False
    pygame.mixer.music.stop()
    is_audio_playing = False
    root.destroy()  # Close the Tkinter window

# Function to handle the window closing event
def on_closing():
    stop_visualizer()

# Create the main application window
root = tk.Tk()
root.title("Music Visualizer")
root.protocol("WM_DELETE_WINDOW", on_closing)  # Bind window closing event to on_closing function

# Set dark theme for the visualization
root.configure(bg="black")
canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_HEIGHT, bg="black")
canvas.pack()

# Create a Progress Bar
progress_bar = canvas.create_rectangle(0, WINDOW_HEIGHT - PROGRESS_BAR_HEIGHT, 0, WINDOW_HEIGHT, fill="white")

# Create a button to start the visualizer
start_button = tk.Button(root, text="Start Visualizer", command=start_visualizer)
start_button.pack()

# Create a button to stop the visualizer
stop_button = tk.Button(root, text="Stop Visualizer", command=stop_visualizer)
stop_button.pack()

# Start the Tkinter main loop
root.mainloop()

