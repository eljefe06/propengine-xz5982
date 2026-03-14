#!/usr/bin/env bash
# Penpot MCP Server - HTTP mode on port 3100
set -a
source "$(dirname "$0")/.env.penpot"
set +a

exec node node_modules/@zcubekr/penpot-mcp-server/dist/index.js
