#!/usr/bin/env bash
# Installs every tool the project needs on macOS (Apple Silicon or Intel) via Homebrew.
# Safe to re-run. Docker Desktop must be started manually after the first install.
set -euo pipefail

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it first:  https://brew.sh" >&2
  exit 1
fi

brew_install()      { brew list --formula "$1" >/dev/null 2>&1 && echo "[ok] $1" || brew install "$1"; }
brew_install_cask() { brew list --cask "$1"    >/dev/null 2>&1 && echo "[ok] $1" || brew install --cask "$1"; }

brew_install python@3.11
brew_install git
brew_install kubernetes-cli     # kubectl (includes kustomize support)
brew_install helm
brew_install minikube           # optional - only if you do not enable Kubernetes in Docker Desktop
brew_install jq
brew_install_cask docker        # Docker Desktop

cat <<MSG

Done. Remaining manual steps:
  1. Start Docker Desktop (whale icon in the menu bar) and wait until it says "running".
  2. Docker Desktop -> Settings -> Kubernetes -> "Enable Kubernetes" -> Apply & Restart
     (or skip this and use Minikube:  minikube start --driver=docker)
  3. Optional, for the PDF export of the report:  brew install --cask libreoffice
Then:  make setup && make data && make train
MSG
