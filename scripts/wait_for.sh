#!/usr/bin/env bash
# wait until every argument exists; "run:<name>" waits for a finished training run
# (runs/<name>/done.flag, or the "done" event in its log)
for f in "$@"; do
  if [[ $f == run:* ]]; then
    r=${f#run:}
    until [ -e runs/$r/done.flag ] || grep -q '"event": "done"' runs/$r/log.jsonl 2>/dev/null; do sleep 30; done
  else
    until [ -e "$f" ]; do sleep 30; done
  fi
done
