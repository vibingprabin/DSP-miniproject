"""
AudioStreamer module handles real-time audio capture from the microphone.
Uses sounddevice.InputStream for non-blocking capture and a thread-safe Queue
to pass data to the main thread.
"""
import queue
import numpy as np
import config

# sounddevice pulls in PortAudio, which isn't available in headless/CI
# environments (or the offline test suite). Import lazily so the DSP core can
# be used without any audio hardware present.
try:
    import sounddevice as sd
    _SD_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover - depends on host audio stack
    sd = None
    _SD_IMPORT_ERROR = _e

class AudioStreamer:
    def __init__(self, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE, channels=config.CHANNELS):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.dtype = config.AUDIO_DTYPE
        
        self.stream = None
        self.audio_queue = queue.Queue()
        self._is_active = False

    def _audio_callback(self, indata, frames, time, status):
        """Callback invoked by PortAudio in a separate high-priority thread."""
        if status:
            pass # In a real app we might log this, but avoid heavy prints here
        # MUST copy indata because the buffer is reused by PortAudio
        self.audio_queue.put(indata.copy())

    def start_stream(self) -> None:
        """Begin continuous microphone capture."""
        if self._is_active:
            return

        if sd is None:
            raise RuntimeError(
                "sounddevice/PortAudio is not available in this environment "
                f"({_SD_IMPORT_ERROR}). Install it to use live microphone input."
            )

        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
                
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        )
        self.stream.start()
        self._is_active = True

    def stop_stream(self) -> None:
        """Stop capture and release device."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self._is_active = False

    def get_chunk(self) -> np.ndarray | None:
        """Non-blocking: returns next audio chunk from queue, or None."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def record_duration(self, duration: float) -> np.ndarray:
        """Blocking: records `duration` seconds and returns concatenated array."""
        was_active = self._is_active
        if not was_active:
            self.start_stream()
            
        import time
        time.sleep(duration)
        
        if not was_active:
            self.stop_stream()
            
        chunks = []
        while not self.audio_queue.empty():
            try:
                chunks.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
                
        if not chunks:
            return np.zeros(int(self.sample_rate * duration))
            
        return np.concatenate(chunks)

    @property
    def is_active(self) -> bool:
        """Whether the stream is currently capturing."""
        return self._is_active
