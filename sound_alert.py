"""Cross-platform beep alarm with no extra dependency beyond the stdlib.

Generates a short sine-wave tone once, writes it to a temp .wav, and plays it
with whatever system player is available (afplay/paplay/aplay/winsound).
Playback runs in a background thread so it never blocks the camera loop.
"""
import array
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave


def _make_tone(path, freq=880.0, duration=0.35, volume=0.5, rate=44100):
    frame_count = int(rate * duration)
    samples = array.array('h')
    amplitude = int(32767 * volume)
    for i in range(frame_count):
        t = i / rate
        # Fade in/out slightly to avoid a click at the edges of the tone.
        fade = min(1.0, i / (rate * 0.02), (frame_count - i) / (rate * 0.02))
        samples.append(int(amplitude * fade * math.sin(2 * math.pi * freq * t)))
    with wave.open(path, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(samples.tobytes())


class SoundAlert:
    """Plays a repeating beep while armed; stays silent otherwise."""

    def __init__(self, interval_seconds=1.2):
        self.interval_seconds = interval_seconds
        self._tone_path = os.path.join(tempfile.gettempdir(), 'eye_alarm_tone.wav')
        _make_tone(self._tone_path)
        self._armed = threading.Event()
        self._stop = threading.Event()
        self._last_play = 0.0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_armed(self, armed):
        if armed:
            self._armed.set()
        else:
            self._armed.clear()

    def beep_once(self):
        self._play()

    def close(self):
        self._stop.set()
        self._armed.set()  # wake the loop so it can exit promptly

    def _loop(self):
        while not self._stop.is_set():
            if self._armed.is_set():
                now = time.monotonic()
                if now - self._last_play >= self.interval_seconds:
                    self._play()
                    self._last_play = now
                time.sleep(0.05)
            else:
                self._armed.wait(timeout=0.2)

    def _play(self):
        try:
            if sys.platform.startswith('win'):
                import winsound
                winsound.PlaySound(self._tone_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                return
            player = 'afplay' if sys.platform == 'darwin' else None
            if player is None:
                for candidate in ('paplay', 'aplay'):
                    if subprocess.run(['which', candidate], capture_output=True).returncode == 0:
                        player = candidate
                        break
            if player:
                subprocess.Popen([player, self._tone_path],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print('\a', end='', flush=True)  # terminal bell fallback
        except Exception:
            print('\a', end='', flush=True)
