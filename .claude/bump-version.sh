#!/bin/bash
# Auto-bump patch version when plugin files are modified.
# Runs once per Claude session per plugin to avoid over-incrementing.

INPUT=$(cat)
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // ""')
SESSION=$(echo "$INPUT" | jq -r '.session_id // "nosession"')

# Determine which plugin was touched
ENTRY=""
LOCK=""
if [[ "$FILE" == */myrock-mail-engine/* ]]; then
    ENTRY="/home/user/propengine-xz5982/myrock-mail-engine/myrock-mail-engine.php"
    LOCK="/tmp/mrme_bumped_${SESSION}"
elif [[ "$FILE" == */myrock-license-server/* ]]; then
    ENTRY="/home/user/propengine-xz5982/myrock-license-server/myrock-license-server.php"
    LOCK="/tmp/mrls_bumped_${SESSION}"
else
    exit 0
fi

# Bump only once per session per plugin
[ -f "$LOCK" ] && exit 0

# Read current patch version from plugin header
CURRENT=$(grep -oP '(?<=\* Version:\s{11})\d+\.\d+\.\d+' "$ENTRY" 2>/dev/null)
[ -z "$CURRENT" ] && exit 0

IFS='.' read -ra V <<< "$CURRENT"
NEW="${V[0]}.${V[1]}.$((V[2] + 1))"

# 1. Bump header comment line
sed -i "s/\(\* Version:\s*\)${CURRENT}/\1${NEW}/" "$ENTRY"

# 2. Bump define() constants (any MR*_VERSION constant on that line)
sed -i "/define.*VERSION/s/'${CURRENT}'/'${NEW}'/" "$ENTRY"

touch "$LOCK"

echo "version bump: ${CURRENT} → ${NEW}  (${ENTRY##*/})"
