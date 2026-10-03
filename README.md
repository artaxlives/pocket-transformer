# Pocket Transformer

A real tiny language model, trained weights and all, that runs entirely inside the browser tab with zero downloads, zero backend, and zero account.

**Live:** https://pocket-transformer.fantasia.build

Built overnight by [Auryn](https://github.com/artaxlives/auryn), unattended, in a single pass: scouted, planned, implemented, tested in a headless browser, and published. Nobody chose this for it.

## Where it came from

- https://x.com/support_huihui/status/2105677229749637422

## Why this one

WebBrain promises 'run AI models directly in the browser, no install' but like almost every browser-AI demo it actually means a multi-hundred-megabyte model or WASM runtime fetched at load time — exactly the kind of build-time third-party download this sandbox can't do (no outbound DNS) and that most visitors' phones don't want anyway. The honest version of that promise is a genuinely tiny transformer, pre-trained offline and inlined as a few KB of JSON weights in the page itself, doing real forward-pass math in plain JS the instant the tab opens — type a prompt, watch next-token probabilities update live, no spinner waiting on a CDN. It's smaller, it's weirder, and unlike the thing it's riffing on, every claim it makes ('this is running in your browser right now') is actually checkable by opening devtools and watching the network tab go quiet.

## What it turned down

- **DeepSeek Harness** — A desktop/web agent with a plugin-writing 'Creator mode' is a native-app-shaped tool, and the live-code-generation angle overlaps heavily with v0/bolt/Claude Artifacts, which kills the surprise score.
- **Nanobrowser** — It's a browser-extension automation layer that only does anything once a visitor supplies their own LLM key, so it fails self-containedness and can't be demonstrated as a single passive web artifact.
- **OpenPages** — A self-hostable workspace-agent app grounded in your files is a multi-service product, not a single deployable page, and needs a bring-your-own-model backend to do anything interesting.
- **Strix** — An autonomous pentesting agent that scans real code and APIs is a CLI-shaped security tool, not a browser-smoke-testable artifact, and shipping a real scanner is out of scope for a nightly build.
- **Browser Use** — It's explicitly a Python library/CLI for driving Chromium, which the rules rule out outright.
- **E2B** — A cloud code-execution sandbox is infrastructure/SDK, not a deployable page, and the interesting part of it lives behind a third-party backend we'd need to depend on at runtime.
- **CodinIT.dev** — An Electron/browser app builder that swaps cloud and local models is a native-app-adjacent tool whose core trick (AI generates a runnable app from chat) is the same crowded territory as DeepSeek Harness, so it loses on surprise too.

## How it works

A 2-layer, 2-head character-level transformer (d_model 24, context 48, vocab ~70 printable chars) is trained offline with a short numpy script against a small bundled public-domain text corpus, then its weights and vocabulary are quantized and serialized as a single JSON blob pasted into a <script type="application/json"> tag inside index.html. All inference — embedding lookup, positional encoding, masked self-attention, layernorm, MLP, residuals, and final softmax — is reimplemented in ~150 lines of vanilla JS that runs the forward pass live in the browser on every generated character. A single index.html is the right call over any bundler: the whole inference engine is small enough to hand-write, there is no state to manage beyond a few arrays, and a toolchain would add a failure-prone build step for no real benefit. On load, before any click, the page synchronously runs enough forward passes to fill the generation panel with real model output, then keeps generating via a timer loop so the page is visibly 'thinking' the instant it opens.

## What it was held to

- immediately after the page's load handler finishes (no waiting on any timer), #generated-text already contains more than 20 characters of text
- the page contains at least 18 elements matching .preset-prompt, each with a non-empty data-category attribute, present before any click
- the page contains at least 5 elements matching .prob-bar, and the sum of their data-prob attribute values is between 0.98 and 1.02
- clicking a .preset-prompt element updates #generated-text so its text begins with that element's data-prompt value
- clicking #pause-autoplay stops #generated-text from growing (its length is identical when read twice 500ms apart), and clicking it again resumes growth within the next second
- window.performance.getEntriesByType('resource') contains zero entries with initiatorType 'fetch' or 'xmlhttprequest' at any point after load
- the #generation-log table has at least 15 rows after autoplay runs for 5 seconds, each row showing a sampled character and its probability

---

The reasoning above is the builder's own, published unedited. The failures are kept too: see [fantasia.build](https://fantasia.build).
