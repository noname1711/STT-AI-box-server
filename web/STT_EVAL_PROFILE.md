# STT evaluation capture profile

This package keeps the existing UI and WebSocket behavior. Only the microphone
capture path is tuned for controlled STT evaluation:

- requests mono 16 kHz input when supported;
- requests echo cancellation OFF;
- requests noise suppression OFF;
- requests automatic gain control OFF;
- requests a 16 kHz AudioContext so browser-native resampling happens before
  the AudioWorklet;
- uses a sample-preserving worklet fast path when the AudioContext is 16 kHz;
- keeps the existing 20 ms / 320-sample PCM16 frame contract;
- keeps the existing no-silent-drop transport behavior.

For a controlled test, use the same microphone, distance, room, browser and
input level for every comparison. After microphone start, confirm the Audio
line reports `AudioContext 16000 Hz` and does not report `resample fallback`.
The browser may still expose hardware/OS DSP that JavaScript cannot disable.
