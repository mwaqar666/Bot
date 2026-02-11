import os
import zipfile


def create_colab_package():
    # Define what to include
    base_dir = "ai_bot"
    files_to_zip = []

    # 1. Walk through ai_bot directory
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__ and logs
        if "__pycache__" in root or "tensorboard_logs" in root:
            continue

        for file in files:
            if file.endswith(".pyc") or file.endswith(".zip"):
                continue

            # Get full local path
            full_path = os.path.join(root, file)
            # Tuple: (local_path, archive_name)
            # archive_name should keep structure e.g. ai_bot/rl_env/trading_env.py
            files_to_zip.append((full_path, full_path))

    # 2. Add config.py (it's in root)
    if os.path.exists("config.py"):
        files_to_zip.append(("config.py", "config.py"))

    # 3. Add .env (optional, but good for keys if needed, though safely ignored usually)
    # Let's skip .env to stay safe.

    zip_filename = "crypto_bot_colab.zip"

    print(f"Creating {zip_filename} with {len(files_to_zip)} files...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for local_file, archive_name in files_to_zip:
            print(f"  Adding {archive_name}")
            zipf.write(local_file, archive_name)

    print(f"\nSUCCESS! {zip_filename} created.")
    print("Next Steps:")
    print("1. Open Google Colab (https://colab.research.google.com)")
    print("2. Upload 'crypto_bot_colab.zip'")
    print("3. Run the installation and training cells.")


if __name__ == "__main__":
    create_colab_package()
