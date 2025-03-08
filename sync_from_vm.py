import os
import subprocess
import argparse

# VM Connection Details
VM_IP = "158.37.65.122"  # Replace with your actual VM IP
VM_USER = "ubuntu"
SSH_KEY = "~/VM/mobl.pem" 

# Local & Remote Paths
LOCAL_EPISODE_PATH = "MoJo/hierarchical/data/episodes"  # Replace with your local episode destination
LOCAL_WEIGHTS_PATH = "../VM/weights"  # Replace with your local weights destination
REMOTE_EPISODE_PATH = "/home/ubuntu/Lux-Design-S3/MoJo/hierarchical/data/episodes"  # No trailing "/"
REMOTE_WEIGHTS_PATH = "Lux-Design-S3/MoJo/hierarchical/weights"  # No trailing "/"

# Ensure the local directories exist
os.makedirs(LOCAL_EPISODE_PATH, exist_ok=True)
os.makedirs(LOCAL_WEIGHTS_PATH, exist_ok=True)

def test_connection():
    """Check SSH connection to VM before syncing."""
    print("🔍 Testing VM connection...")

    ssh_test_cmd = ["ssh", "-i", SSH_KEY, f"{VM_USER}@{VM_IP}", "echo Connection Successful"]
    try:
        result = subprocess.run(ssh_test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if "Connection Successful" in result.stdout:
            print("✅ SSH connection to VM is successful!")
            return True
        else:
            print("⚠️ SSH connection test failed! Check your SSH key, IP address, or VM status.")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("❌ SSH connection timed out! Ensure the VM is running and accessible.")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ SSH connection error: {e}")
        return False

def sync_episode(episode_id):
    """Sync a specific episode from VM to local machine and always sync weights."""
    print(f"📂 Fetching episode_{episode_id}.json from VM...")

    rsync_episode_command = [
        "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
        f"{VM_USER}@{VM_IP}:{REMOTE_EPISODE_PATH}/episode_{episode_id}.json",
        LOCAL_EPISODE_PATH
    ]

    rsync_weights_command = [
        "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
        f"{VM_USER}@{VM_IP}:{REMOTE_WEIGHTS_PATH}/",
        LOCAL_WEIGHTS_PATH
    ]

    try:
        subprocess.run(rsync_episode_command, check=True)
        print(f"✅ Episode {episode_id} synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error syncing episode {episode_id}: {e}")

    # Always sync weights
    try:
        subprocess.run(rsync_weights_command, check=True)
        print("✅ Weights synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error syncing weights: {e}")

def sync_all():
    """Sync all episodes and weights from VM to local machine."""
    print("📂 Syncing all episodes and weights from VM...")

    rsync_episode_command = [
        "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
        f"{VM_USER}@{VM_IP}:{REMOTE_EPISODE_PATH}/",
        LOCAL_EPISODE_PATH
    ]

    rsync_weights_command = [
        "rsync", "-avz", "-e", f"ssh -i {SSH_KEY}",
        f"{VM_USER}@{VM_IP}:{REMOTE_WEIGHTS_PATH}/",
        LOCAL_WEIGHTS_PATH
    ]

    try:
        subprocess.run(rsync_episode_command, check=True)
        print("✅ All episodes synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error syncing episodes: {e}")

    try:
        subprocess.run(rsync_weights_command, check=True)
        print("✅ Weights synced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error syncing weights: {e}")

def main():
    """Main function to handle sync process when called manually."""
    parser = argparse.ArgumentParser(description="Sync weights and episodes from VM to local machine.")
    parser.add_argument("--id", type=int, help="Specify episode ID to fetch (e.g., --id 10).")
    args = parser.parse_args()

    if not test_connection():
        print("❌ Unable to connect to VM. Exiting.")
        return

    if args.id:
        sync_episode(args.id)  # Fetch only the specified episode, but always sync weights
    else:
        sync_all()  # Sync everything by default

if __name__ == "__main__":
    main()
