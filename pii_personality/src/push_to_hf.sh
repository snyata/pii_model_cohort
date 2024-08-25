#!/bin/bash

# Function to check if already authenticated
check_authentication() {
    if hf_user_info=$(huggingface-cli whoami 2>&1); then
        echo "Already authenticated as: $hf_user_info"
        return 0
    else
        echo "Not authenticated. Please log in."
        return 1
    fi
}

# Function to log in
login_huggingface() {
    echo "Logging in to Hugging Face..."
    huggingface-cli login
}

# Function to push the file to Hugging Face Hub
push_to_hub() {
    local file_path="$1"
    local repo_name="$2"

    if [ -z "$file_path" ] || [ -z "$repo_name" ]; then
        echo "Usage: push_to_hub <file_path> <repo_name>"
        exit 1
    fi

    echo "Pushing $file_path to Hugging Face Hub in repo $repo_name..."
    huggingface-cli upload "$file_path" --repo-id "$repo_name"
}

# Main script
if ! check_authentication; then
    login_huggingface
fi

# Example usage: pass the file path and repo name as arguments
push_to_hub "$1" "$2"