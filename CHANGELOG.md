# Changelog

## 0.1.0 — milestone 1: foundation (unreleased)

- App shell: navigator, Jobs and Log docks, undo/redo through project edits, command palette (Ctrl+K), dark and light
  themes, status bar (server, GPU, memory).
- Project format version 1 with JSON Schema, migration from the pre-release version 0, validation on load and save.
- Job queue with a machine-wide CARLA lock, a GPU lock and a memory guard; QProcess launcher (GUI) and subprocess
  launcher (CLI) over the same specs and protocol.
- CARLA server manager: headless, memory-capped systemd scope, NVIDIA Vulkan ICD pinned, own ports, never a
  user's server or CARLA's default ports unless chosen.
- Workers (probe, tidy, drive) with signals blocked before numpy, a watchdog, run.json and exit codes.
- Drive: synchronous fixed step, seeded traffic, cameras with semantic label cameras at the same pose and tick.
- Data page (record, browse with label overlay and legend), Worlds page (towns, weather), Home, Settings, Help.
- First-run wizard with a 30 s smoke drive that records the machine's throughput.
- CLI: doctor, project, probe, smoke, drive, tidy, reproduce, runs, schema.
- Found and fixed on a real 0.9.16 server: `ActorAttribute` has no `.value` (use `as_str()`); destroying the ego
  before its sensors, ticking after destroying traffic-manager vehicles, or `Actor.destroy()` on any vehicle that was
  on the autopilot ends the client process with `std::terminate` — vehicles are now destroyed by id in a batch.
