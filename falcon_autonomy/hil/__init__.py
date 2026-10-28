"""Hardware in the loop: a device (a Jetson AGX Orin, or a simulated one on this machine) receives a CARLA session's
camera and lidar in real time, runs a stack and sends control back; every frame's latency is measured stage by stage.

Three processes, each in the Python that has what it needs (docs/hardware.md is the guide):

* the session worker (falcon_autonomy.workers.hil) in CARLA's Python: steps the world in synchronous mode at a fixed
  rate paced to the wall clock, reads one camera and one lidar, stamps them, hands them to the gateway and applies
  the control that came back at the next step;
* the gateway (falcon_autonomy.hil.gateway) in the app's Python (GStreamer through PyGObject, which CARLA's venv does
  not have): encodes the camera (H.264 over RTP/UDP, or raw over TCP), frames the lidar (TCP or UDP), talks to the
  device, measures the clock offset to it, and passes the control back;
* the device agent (falcon_autonomy.hil.device_agent): one file, Python ≥ 3.10 with the standard library, numpy and
  (for H.264) python3-gi — what JetPack 6 has — receiving the streams, running a pluggable stack and sending control.

core.hil holds the configs, the emulated sensor clocks, the clock-offset estimate and the latency statistics;
hil.report the waterfall and the report; hil.deploy copying and starting the agent over SSH with the user's key.
"""
