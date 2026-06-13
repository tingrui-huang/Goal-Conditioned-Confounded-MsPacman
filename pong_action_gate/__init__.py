"""Pong action-sensitivity gate (CCRL Phase 1).

The teacher (`ActorCritic`, preprocessing, checkpoint utils) is imported directly
and unmodified from the PRIVATE `Confounded-Agent-Distillation` repo, treated as a
read-only external dependency. Configure its location with the env var
`CONF_AGENT_TEACHER_PATH`; see `teacher/external_teacher.py`. No teacher code is
copied or committed.
"""
