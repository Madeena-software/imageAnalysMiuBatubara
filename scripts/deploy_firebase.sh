#!/usr/bin/env bash
set -euo pipefail

# Deploy script for Firebase Hosting
# Usage: `bash scripts/deploy_firebase.sh`
# Requires: either `firebase` CLI logged-in, or a `FIREBASE_TOKEN` env var

PROJECT_ID="project-cc2e5"

if command -v firebase >/dev/null 2>&1; then
  echo "Using installed firebase CLI"
  firebase deploy --only hosting --project "$PROJECT_ID"
else
  echo "Using npx to run firebase-tools (will download if needed)"
  npx --yes firebase-tools deploy --only hosting --project "$PROJECT_ID"
fi
