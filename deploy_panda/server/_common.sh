# Shared bootstrap for the scripts in this directory. Source it, do not execute it.

set -euo pipefail

DEPLOY_PANDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$DEPLOY_PANDA_DIR/.." && pwd)"

if [[ -f "$DEPLOY_PANDA_DIR/config.sh" ]]; then
  # shellcheck disable=SC1091
  source "$DEPLOY_PANDA_DIR/config.sh"
else
  echo "note: $DEPLOY_PANDA_DIR/config.sh not found; using defaults + environment" >&2
  # shellcheck disable=SC1091
  source "$DEPLOY_PANDA_DIR/config.example.sh"
fi

# Fail fast with a pointer to the setting, rather than acting on an empty path.
require_var() {
  local name="$1" hint="${2:-}"
  if [[ -z "${!name:-}" ]]; then
    echo "error: $name is not set." >&2
    echo "       Set it in $DEPLOY_PANDA_DIR/config.sh, or pass it inline: $name=... $0" >&2
    [[ -n "$hint" ]] && echo "       $hint" >&2
    exit 1
  fi
}
