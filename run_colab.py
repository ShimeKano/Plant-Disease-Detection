import os
import re
import subprocess
import time

REPO_DIR = "/content/Plant-Disease-Detection"
REPO_URL = "https://github.com/ShimeKano/Plant-Disease-Detection.git"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "-q", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)

subprocess.run(
    ["python", "-m", "pip", "install", "-q", "-r", "requirments.txt"],
    check=True,
)
subprocess.run(
    ["python", "-m", "pip", "install", "-q", "streamlit", "cloudflared"],
    check=True,
)

subprocess.run(["pkill", "-f", "streamlit run"], check=False)
subprocess.run(["pkill", "-f", "cloudflared"], check=False)

with open("/content/streamlit.log", "w") as log:
    subprocess.Popen(
        [
            "streamlit",
            "run",
            "app.py",
            "--server.address",
            "0.0.0.0",
            "--server.port",
            "8501",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )

with open("/content/tunnel.log", "w") as log:
    subprocess.Popen(
        [
            "cloudflared",
            "tunnel",
            "--url",
            "http://127.0.0.1:8501",
            "--no-autoupdate",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )

print("🌱 Plant Disease AI đang khởi động...")

url = None
for _ in range(20):
    time.sleep(1)
    try:
        with open("/content/tunnel.log", encoding="utf-8") as f:
            log = f.read()
        matches = re.findall(r"https://[^\s]+\.trycloudflare\.com", log)
        if matches:
            url = matches[-1]
            break
    except FileNotFoundError:
        pass

if url:
    print(f"🌐 Website: {url}")
else:
    print("⚠️ Chưa lấy được Cloudflare URL.")
    print("Kiểm tra log bằng: !cat /content/tunnel.log")

print("🟢 Streamlit chạy tại port 8501.")
print("ℹ️ Colab runtime phải tiếp tục hoạt động để website duy trì.")
