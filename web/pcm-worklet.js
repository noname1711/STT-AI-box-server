class HLPcm16Worklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};
    this.targetRate = Number(opts.targetSampleRate || 16000);
    this.frameSamples = Number(opts.frameSamples || 320);

    this.inputRate = sampleRate;
    this.step = this.inputRate / this.targetRate;

    this.input = [];
    this.readPos = 0;
    this.output = [];

    // PHASE6G_VI_FINAL_TAIL_FLUSH
    //
    // Control-plane message from app.js. The app invokes this only for
    // Vietnamese capture. Emit the exact remaining PCM samples; do not pad
    // to 20 ms. meeting-server already accepts arbitrary packet lengths and
    // owns the final PCM remainder flush.
    this.port.onmessage = (event) => {
      const msg = event?.data || {};
      if (msg.type !== "flush") return;

      const remaining = this.output.length;
      if (remaining > 0) {
        this.emitFrame(this.output.splice(0, remaining));
      }

      // postMessage ordering guarantees the PCM buffer is queued before ACK.
      this.port.postMessage({
        type: "flush_ack",
        samples: remaining,
      });
    };
  }

  process(inputs) {
    const channels = inputs[0];
    if (!channels || !channels[0] || channels[0].length === 0) return true;

    const mono = channels[0];

    // Accuracy-first fast path. app.js requests a 16 kHz AudioContext, so
    // samples arrive here already resampled by the browser audio engine.
    // Preserve them sample-for-sample until PCM16 quantization.
    if (this.inputRate === this.targetRate) {
      for (let i = 0; i < mono.length; i++) {
        this.output.push(mono[i]);
        if (this.output.length >= this.frameSamples) {
          this.emitFrame(this.output.splice(0, this.frameSamples));
        }
      }
      return true;
    }

    // Compatibility fallback for browsers that ignore AudioContext.sampleRate.
    // This path is visible in the UI/event log and should not be used for
    // controlled STT accuracy comparisons when a native 16 kHz context works.
    for (let i = 0; i < mono.length; i++) this.input.push(mono[i]);

    while (this.readPos + 1 < this.input.length) {
      const i0 = Math.floor(this.readPos);
      const frac = this.readPos - i0;
      const a = this.input[i0];
      const b = this.input[i0 + 1];
      const sample = a + (b - a) * frac;
      this.output.push(sample);
      this.readPos += this.step;

      if (this.output.length >= this.frameSamples) {
        this.emitFrame(this.output.splice(0, this.frameSamples));
      }
    }

    const consumed = Math.floor(this.readPos);
    if (consumed > 0) {
      this.input.splice(0, consumed);
      this.readPos -= consumed;
    }
    return true;
  }

  emitFrame(frame) {
    const buffer = new ArrayBuffer(frame.length * 2);
    const view = new DataView(buffer);
    for (let i = 0; i < frame.length; i++) {
      const x = Math.max(-1, Math.min(1, frame[i]));
      const s = x < 0 ? Math.round(x * 32768) : Math.round(x * 32767);
      view.setInt16(i * 2, s, true);
    }
    this.port.postMessage(buffer, [buffer]);
  }
}

registerProcessor("hl-pcm16-worklet", HLPcm16Worklet);
