# Plan: Pluggable Realtime Backends + Barge-in

Project goal: keep this project's Voice PE → local server → realtime LLM architecture,
add **true barge-in** (currently half-duplex), make the realtime backend **pluggable**
(OpenAI Realtime and Ultravox), and move HA tool access fully onto the LAN.

## Codebase findings

### Firmware (`home-assistant-voice-pe/`)
- Custom ESPHome component `voice_assistant_websocket`.
- Protocol is minimal: binary frames = raw PCM (no headers); text frames =
  `{"type":"interrupt"}` and `{"type":"disconnect"}`.
- Device → server: mic 16kHz/32-bit/stereo (shared with microWakeWord) converted
  in code to 24kHz/16-bit/mono. Server → device: 24kHz/16-bit/mono → ESPHome
  resampler → 48kHz speaker.
- Flush-on-interrupt already exists: server-sent `interrupt` stops speaker and
  clears the audio queue (`voice_assistant_websocket.cpp`).
- Half-duplex today: mic audio is dropped while `is_bot_speaking()` (audio
  received within 500ms).
- Client-initiated `interrupt()` exists (wake-word triggered only).
- XMOS `voice_kit`: AEC always active in hardware; AGC (ch0) / NS (ch1).
- ESPHome `min_version: 2025.11.0`; voice_kit ext component ref `25.11.0`;
  XMOS firmware v1.3.1.

### Server (`openai_realtime_voice_agent/`)
- Built on **pipecat** (`pipecat-ai[mcp,openai,websocket] >= 0.0.96`).
- `WebsocketServerTransport` + custom `RawAudioSerializer` (binary ⇄ raw PCM,
  24kHz hardcoded).
- `OpenAIRealtimeLLMService` with `server_vad` turn detection; context
  aggregator; `SessionManager` reuse (300s).
- HA tools via MCP: `HomeAssistantMCPService` → pipecat `MCPClient`
  (StreamableHTTP + Bearer token) against HA `/api/mcp`; tool schemas converted
  to function defs and registered as handlers.
- Model-invoked `disconnect` tool ends the session.
- Known issue: default `HA_MCP_URL` is addon-specific; for Docker HA use
  `http://<ha-host>:8123/api/mcp` + long-lived token.

## Principles

1. Each phase independently testable; device must keep working after each step.
2. Backward compatible protocol: old server + new device (and vice versa) still work.
3. Firmware contains **zero backend knowledge** — the server owns the audio format.
4. Barge-in is backend-agnostic and a separately PR-able improvement.
5. Pin dependencies (pipecat churns fast); treat upstream as reference, not dependency.

## Phases

### Phase 0 — Setup & verification
- [x] Clone repo; read firmware + server code.
- [x] Stock Voice PE onboarded to HA (no pipeline).
- [x] Rehearse USB re-flash of stock firmware via web.esphome.io (Chrome;
      hold button while plugging if device doesn't enumerate). Re-adopt in HA.
      Verified: connect to "USB JTAG/serial debug unit" port → Install →
      upload factory bin from esphome/home-assistant-voice-pe releases
      (no project cards on web.esphome.io); re-provision Wi-Fi via
      Configure Wi-Fi over USB (improv); HA Reconfigure re-negotiates the
      API encryption key.
- [x] Decide proxy runtime: standalone Docker container on LAN box.
      Decided: host-networked compose service on the x86_64 LAN box, port
      8090 (8080 taken by zigbee2mqtt). Build with
      BUILD_FROM=ghcr.io/home-assistant/amd64-base; bypass the addon
      run.sh (bashio::config needs Supervisor) via
      `command: python3 -m app.main` + env vars. HA MCP at
      http://127.0.0.1:8123/api/mcp with long-lived token (secrets.env:
      OPENAI_API_KEY, LONGLIVED_TOKEN).

### Phase 1 — Unmodified stack end-to-end on OpenAI
- [ ] Build + flash fork firmware as-is (`secrets.yaml` with `server_url`).
      Build done: compiles clean under esphome 2025.11.5 (docker image);
      `home-assistant-voice-pe/ha-voice-openai.factory.bin` ready to flash
      via web.esphome.io → Connect → Install → upload file.
      `secrets.yaml` created with generated api/ota keys and
      server_url ws://192.168.1.2:8090; wifi creds are placeholders —
      fill them in before flashing, or provision via BLE improv /
      USB (Configure Wi-Fi) afterwards.
      Flash pending (device + laptop needed).
- [x] Run server unchanged (Docker) against HA MCP with long-lived token.
      Done: standalone host-networked container (`voice-agent` in the
      homeautomation compose), WS listening on 0.0.0.0:8090, HA MCP at
      127.0.0.1:8123/api/mcp; 20 Assist tools fetched and registered.
      Note: HA `mcp_server` integration had to be added once (config entry
      "Assist") before `/api/mcp` stopped 404-ing.
- [ ] **Milestone:** wake word → conversation → HA commands work (half-duplex).

### Phase 2 — Barge-in behind a flag (default off = Phase 1 behavior)
- [ ] Protocol: hello/config message on connect, designed once to carry both
      format and `barge_in`:
      `{"type":"config","input":{"sample_rate":16000,"format":"pcm_s16le","channels":1},
        "output":{...},"barge_in":true}`
- [ ] Firmware: stream mic during playback only when server advertises barge-in
      (local `allow_barge_in` YAML flag as fallback for old servers).
- [ ] Server: on backend interruption signal (OpenAI server VAD → cancel
      response), send `{"type":"interrupt"}` to device; rely on pipecat
      interruption frames where possible.
- [ ] **Milestone:** flag off = identical to Phase 1; flag on = talk over agent,
      speaker stops ≤200ms, conversation continues coherently.

### Phase 3 — Pluggable backend via server-side descriptors
- [ ] Backend descriptor config (YAML/TOML) per backend: service factory,
      turn-detection config, tool wiring, audio contract
      (rates/bit-depth/channels).
- [ ] Device configures resampler/stream-info from hello message; no config
      message → current 24kHz behavior (backward compat).
- [ ] Firmware input path: pass-through when server rate == mic rate (16kHz),
      else resample.
- [ ] **Milestone:** same device, switch backend via server config, no reflash.

### Phase 4 — Ultravox backend
- [ ] Ultravox API contract: confirm realtime WS endpoint, audio format
      (expect 16kHz/16-bit/mono), client-side `function` tool semantics,
      interruption events, session reuse/keep-alive.
- [ ] Backend descriptor + pipecat `UltravoxRealtimeLLMService` integration;
      client-side `function` tools only (no inbound exposure of HA).
- [ ] Session strategy: long-lived Ultravox sessions vs. OpenAI-style
      `SessionManager` context caching — degrade gracefully per backend.
- [ ] **Milestone:** full conversation + HA tools via Ultravox; measure latency
      vs Phase 1 baseline.

### Phase 5 — Hardening
- [ ] Latency tuning (session warmth, buffer sizes, TTS voice).
- [ ] Failure behavior: proxy/backend down → LED + audio cue; document fallback
      to stock firmware.
- [ ] Security review: least-privilege HA token, no inbound exposure, mute
      switch preserved.
- [ ] Upstream: split work into PRs — (1) config handshake, (2) barge-in flag,
      (3) Ultravox backend descriptor.

## Risks / open questions
- Firmware builds against esphome 2025.11.5 (verified); still unverified on
  the hardware itself until the Phase 1 flash test.
- Ultravox interruption event contract + speculative tool behavior (verify via
  Ultravox research, Phase 4).
- `SessionManager` is OpenAI-shaped; needs per-backend strategy (Phase 4).
- pipecat API churn; pin the installed version.
