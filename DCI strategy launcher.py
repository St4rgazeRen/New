# app_launcher.py
import os, sys, time, webbrowser, threading

APP = os.path.join(os.path.dirname(__file__), "DCI strategy.py")
PY  = sys.executable  # 用 Spyder 目前這個 Python

def open_browser(port=8501):
    time.sleep(3)
    webbrowser.open(f"http://localhost:{port}")

if __name__ == "__main__":
    port = 8501  # 若被占用可改 8502/8505
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    cmd = f'"{PY}" -m streamlit run "{APP}" --server.port {port} --server.headless true'
    print("RUN:", cmd)
    os.system(cmd)
