#!/bin/bash

# chmod +x copy_weights_to_vm.sh Make the Script Executable
# ./copy_weights_to_vm.sh Run the Script

# VM Details
VM_IP="158.39.201.251"  # Replace with your VM's actual IP
VM_USER="ubuntu"
SSH_KEY="$HOME/VM/lux.pem"  # Ensure this SSH key is in your current directory or provide the full path

# Source (WSL Path) and Destination (VM Path)
WSL_PATH="$HOME/Lux-Design-S3/MoJo/hierarchical/weights/"
VM_PATH="~/Lux-Design-S3/MoJo/hierarchical/weights/"


echo "🔍 Testing SSH connection to VM..."
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "echo Connection successful"

echo "🔍 Checking if weights directory exists on VM..."
ssh -i "$SSH_KEY" "$VM_USER@$VM_IP" "mkdir -p $VM_PATH"

echo "📂 Copying weights from WSL to VM..."
rsync -avz -e "ssh -i $SSH_KEY" "$WSL_PATH" "$VM_USER@$VM_IP:$VM_PATH"

echo "✅ Weights copied successfully to VM!"
