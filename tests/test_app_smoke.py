import subprocess
import sys
import time


def test_streamlit_app_starts():
    command = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", "8502", "--server.fileWatcherType", "none"]
    proc = subprocess.Popen(command, cwd=".", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(60):
            line = proc.stdout.readline()
            if line:
                if "You can now view your Streamlit app" in line or "Local URL" in line:
                    break
            time.sleep(0.5)
        assert proc.poll() is None or proc.poll() == 0
    finally:
        proc.terminate()
        proc.wait(timeout=10)
