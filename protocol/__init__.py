"""
Three-phase session protocol (v0.4): safety screen -> rest scan ->
guided activity -> transition -> recovery scan -> gates -> heads.

The camera's role per phase is fixed by contract: physiology is measured
ONLY while the subject is still (rest + recovery); during the activity
the camera verifies WORKLOAD (reps/cadence) and nothing else — there is
no code path that estimates HR from a moving subject.
"""
