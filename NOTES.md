## Changed from the plan

No feature or architecture changes. Implemented a single self-contained index.html with inline signed 8-bit quantized weights, a 2-layer / 2-head transformer, width 24, context 48, and a 68-character vocabulary (13,700 parameters). No build step, external assets, runtime dependencies, API calls, or accounts.

Selected the opening of Lewis Carroll's public-domain Alice's Adventures in Wonderland (1865) as the bundled corpus. Trained every parameter with NumPy and Adam for 3,000 steps, batch size 12, seed 42. Training loss decreased from 4.3204 to 0.8782 nats on the final training batch; this is not a held-out quality score. The trainer documents retraining with a replacement corpus. A pre-existing local NumPy installation supplied the offline training environment; the delivered files contain no reference or dependency on that environment.

Added 21 varied prompts covering all requested categories. Startup performs 24 genuine sampled-character steps synchronously, then generates every 140 ms. Every sample is recorded with its original model probability. The chart shows the top six characters plus an explicitly labeled “other” bar so the displayed probability mass sums to 100%. Unknown Unicode characters are preserved in the visible prompt and mapped to spaces for embedding lookup. Layer normalization is non-affine, positional embeddings are learned, and the MLP uses ReLU; these details were unspecified by the plan.

Added small offline verification scripts under tests/. Ran `node tests/verify.cjs` to open and execute the actual HTML script in a minimal DOM harness: initial output exceeded 20 characters, all 21 presets reseeded correctly, bars summed to one, pause held text fixed across simulated timer ticks, resume grew text, and the log contained 66 valid rows after advancing the timer. All asset references resolve; there are no runtime network calls. The displayed initial sample was “Alice was beginning to wonder the how now, down. There”. Sampling is intentionally stochastic.

Ran `python3 tests/reference.py` after the Node harness to compare the quantized JavaScript forward pass with an independent NumPy calculation: maximum probability difference 2.09e-16. NumPy's local macOS matrix backend emitted arithmetic warnings despite finite values and matching results; the trainer explicitly rejects non-finite loss or gradients rather than suppressing failures.

## Could not do

None of the requested functionality was omitted. Model output is deliberately small-corpus character-model text; improving coherence is out of scope.

## Omitted for lack of a service

None. Training, inference, and the interface need no service. No generated images were needed.
