"""
Activity verification (v0.4): the camera's ONLY job during the guided
activity is workload verification — rep/cadence counting from body
motion. There is no code path here that touches pulse, HR, or any
physiological signal; in-motion rPPG fails (13-42 bpm errors in the
literature) and this repo never attempts it.
"""
