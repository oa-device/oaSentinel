#!/bin/bash

# oaSentinel Shell Aliases Setup
# Creates convenient shell aliases for oaSentinel management
# Usage: ./setup_shell_aliases.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

# Detect shell
SHELL_RC=""
if [ -n "$BASH_VERSION" ]; then
    SHELL_RC="$HOME/.bashrc"
elif [ -n "$ZSH_VERSION" ]; then
    SHELL_RC="$HOME/.zshrc"
elif [ "$SHELL" = "/usr/bin/fish" ] || [ "$SHELL" = "/bin/fish" ]; then
    SHELL_RC="$HOME/.config/fish/config.fish"
else
    # Default to bashrc
    SHELL_RC="$HOME/.bashrc"
fi

log_info "Setting up oaSentinel aliases for shell: $SHELL"
log_info "Shell configuration: $SHELL_RC"

# Create aliases based on shell type
if [[ "$SHELL_RC" == *"fish"* ]]; then
    # Fish shell aliases
    ALIASES_CONTENT="
# oaSentinel Aliases (Fish Shell)
alias oas-info='cd $PROJECT_ROOT && source .venv/bin/activate && echo \"🎯 oaSentinel Environment Status\" && echo \"Project: $PROJECT_ROOT\" && echo \"Python: \$(python --version)\" && echo \"GPU: \$(python -c \"import torch; print(f\\\"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\\\")\" 2>/dev/null || echo \"N/A\")\" && ./scripts/detect_dataset.sh --format summary 2>/dev/null || echo \"Dataset: Not configured\"'
alias oas-quick-train='cd $PROJECT_ROOT && screen -dmS oasentinel-training bash -c \"source .venv/bin/activate && ./start_training_auto.sh; exec bash\" && echo \"🚀 Training started in screen session \\\"oasentinel-training\\\"\" && echo \"📺 Attach with: screen -r oasentinel-training\"'
alias oas-screen-attach='screen -r oasentinel-training'
alias oas-screen-list='screen -list | grep oasentinel || echo \"No oaSentinel screen sessions found\"'
alias oas-logs='cd $PROJECT_ROOT && tail -f logs/latest_training.log 2>/dev/null || echo \"No training log found\"'
alias oas-status='cd $PROJECT_ROOT && [ -f logs/last_training_status.txt ] && echo \"Last training status:\" && cat logs/last_training_status.txt || echo \"No training status available\"'
alias oas-setup='cd $PROJECT_ROOT && ./setup.sh'
alias oas-process-data='cd $PROJECT_ROOT && ./scripts/process_data.sh'
alias oas-download-data='cd $PROJECT_ROOT && ./scripts/download_data.sh'
alias oas-export='cd $PROJECT_ROOT && ./scripts/export.sh'
alias oas-cd='cd $PROJECT_ROOT'
"
else
    # Bash/Zsh aliases
    ALIASES_CONTENT="
# oaSentinel Aliases
alias oas-info='cd $PROJECT_ROOT && source .venv/bin/activate && echo \"🎯 oaSentinel Environment Status\" && echo \"Project: $PROJECT_ROOT\" && echo \"Python: \$(python --version)\" && echo \"GPU: \$(python -c \"import torch; print(f\\\"CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}\\\")\" 2>/dev/null || echo \"N/A\")\" && ./scripts/detect_dataset.sh --format summary 2>/dev/null || echo \"Dataset: Not configured\"'
alias oas-quick-train='cd $PROJECT_ROOT && screen -dmS oasentinel-training bash -c \"source .venv/bin/activate && ./start_training_auto.sh; exec bash\" && echo \"🚀 Training started in screen session \\\"oasentinel-training\\\"\" && echo \"📺 Attach with: screen -r oasentinel-training\"'
alias oas-screen-attach='screen -r oasentinel-training'
alias oas-screen-list='screen -list | grep oasentinel || echo \"No oaSentinel screen sessions found\"'
alias oas-logs='cd $PROJECT_ROOT && tail -f logs/latest_training.log 2>/dev/null || echo \"No training log found\"'
alias oas-status='cd $PROJECT_ROOT && [ -f logs/last_training_status.txt ] && echo \"Last training status:\" && cat logs/last_training_status.txt || echo \"No training status available\"'
alias oas-setup='cd $PROJECT_ROOT && ./setup.sh'
alias oas-process-data='cd $PROJECT_ROOT && ./scripts/process_data.sh'
alias oas-download-data='cd $PROJECT_ROOT && ./scripts/download_data.sh'
alias oas-export='cd $PROJECT_ROOT && ./scripts/export.sh'
alias oas-cd='cd $PROJECT_ROOT'
"
fi

# Replace PROJECT_ROOT with actual path
ALIASES_CONTENT=$(echo "$ALIASES_CONTENT" | sed "s|\$PROJECT_ROOT|$PROJECT_ROOT|g")

# Check if aliases already exist
if grep -q "# oaSentinel Aliases" "$SHELL_RC" 2>/dev/null; then
    log_warning "oaSentinel aliases already exist in $SHELL_RC"
    log_info "To update, remove existing aliases and run this script again"
else
    # Add aliases to shell configuration
    echo "$ALIASES_CONTENT" >> "$SHELL_RC"
    log_success "oaSentinel aliases added to $SHELL_RC"
fi

log_info "Available aliases:"
echo "  oas-info           - Show environment status"
echo "  oas-quick-train    - Start training in background"
echo "  oas-screen-attach  - Attach to training session"
echo "  oas-screen-list    - List oaSentinel screen sessions"
echo "  oas-logs           - Follow training logs"
echo "  oas-status         - Show last training status"
echo "  oas-setup          - Run environment setup"
echo "  oas-process-data   - Process dataset"
echo "  oas-download-data  - Download dataset"
echo "  oas-export         - Export trained models"
echo "  oas-cd             - Change to oaSentinel directory"

log_success "Shell aliases setup complete!"
log_info "Reload your shell or run: source $SHELL_RC"
