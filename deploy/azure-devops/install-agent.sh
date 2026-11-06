#!/usr/bin/env bash
# Install (or reconfigure) the Azure DevOps self-hosted agent for pool k3s-host.
#
#   export AZURE_DEVOPS_EXT_PAT=...          # Agent Pools (read + manage)
#   export ADO_ORG=https://dev.azure.com/aimlabdev
#   deploy/azure-devops/install-agent.sh
set -euo pipefail

ADO_ORG="${ADO_ORG:-https://dev.azure.com/aimlabdev}"
ADO_PROJECT="${ADO_PROJECT:-cpathportal}"
POOL_NAME="${POOL_NAME:-k3s-host}"
AGENT_DIR="${AGENT_DIR:-${HOME}/azdo-agent}"
AGENT_VERSION="${AGENT_VERSION:-4.258.1}"
AGENT_NAME="${AGENT_NAME:-$(hostname)-k3s}"

[[ -n "${AZURE_DEVOPS_EXT_PAT:-}" ]] || {
  echo "Set AZURE_DEVOPS_EXT_PAT" >&2
  exit 1
}

mkdir -p "$AGENT_DIR"
cd "$AGENT_DIR"

if [[ ! -f ./config.sh ]]; then
  echo "==> downloading agent ${AGENT_VERSION}"
  curl -fsSL -o agent.tar.gz \
    "https://download.agent.dev.azure.com/agent/${AGENT_VERSION}/vsts-agent-linux-x64-${AGENT_VERSION}.tar.gz"
  tar xzf agent.tar.gz
  rm agent.tar.gz
fi

if [[ -f .agent ]]; then
  echo "==> agent already configured ($(cat .agent | python3 -c 'import json,sys;print(json.load(sys.stdin).get("agentName","?"))' 2>/dev/null || echo configured))"
else
  echo "==> configuring agent ${AGENT_NAME} for pool ${POOL_NAME}"
  ./config.sh --unattended \
    --url "$ADO_ORG" \
    --auth pat \
    --token "$AZURE_DEVOPS_EXT_PAT" \
    --pool "$POOL_NAME" \
    --agent "$AGENT_NAME" \
    --work _work \
    --replace \
    --acceptTeeEula
fi

# systemd user service so the agent survives logout
SVC="${HOME}/.config/systemd/user/azdo-agent.service"
mkdir -p "$(dirname "$SVC")"
cat >"$SVC" <<EOF
[Unit]
Description=Azure DevOps agent (${AGENT_NAME})
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${AGENT_DIR}
ExecStart=${AGENT_DIR}/run.sh
Restart=always
RestartSec=10
Environment=DOCKER_BUILDKIT=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now azdo-agent.service
# Linger so the user service runs without an interactive login.
loginctl enable-linger "$(id -un)" 2>/dev/null || true
systemctl --user --no-pager --full status azdo-agent.service | head -20
echo "agent installed in ${AGENT_DIR}"
