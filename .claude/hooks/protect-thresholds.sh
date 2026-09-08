#!/bin/sh
# PreToolUse hook (Edit|Write|MultiEdit): block edits to the files that hold validity
# thresholds unless the owner approved THIS edit in chat, signalled by a one-shot
# sentinel file the assistant creates right before the edit. Exit 2 = block, and the
# message on stderr is shown to the assistant. Reads the tool call as JSON on stdin.
INPUT=$(cat)
FILE=$(printf '%s' "$INPUT" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
[ -z "$FILE" ] && exit 0
case "$FILE" in
  */configs/*.yaml|*/capture/ingest.py|*/datasets/schema.py|*/inference/pipeline.py|configs/*.yaml|capture/ingest.py|datasets/schema.py|inference/pipeline.py) ;;
  *) exit 0 ;;
esac
ROOT="${CLAUDE_PROJECT_DIR:-.}"
# The sentinel may live in this repo or, when the session runs from the webapp, in the cardio repo.
for S in "$ROOT/.claude/owner-approved" "$ROOT/../cardio/.claude/owner-approved" "/Users/ravindrasinghbisht/Downloads/cardio/.claude/owner-approved"; do
  if [ -f "$S" ]; then rm -f "$S"; echo "[cardio-guard] owner-approved edit to $FILE (sentinel consumed)" >&2; exit 0; fi
done
cat >&2 <<MSG
[cardio-guard] BLOCKED: $FILE holds validity thresholds (gates, floors, calibration).
Changing it is an owner decision, taken in chat, one threshold at a time, with corpus
evidence (see CLAUDE.md "Rules that are enforced"). If the owner has just approved this
exact edit, run: touch /Users/ravindrasinghbisht/Downloads/cardio/.claude/owner-approved
and retry once. Otherwise improve the signal instead (the optimizer skill).
MSG
exit 2
