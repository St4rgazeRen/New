# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 23:13:08 2025

@author: winte
"""

# app_launcher.py
import os, sys, subprocess
from pathlib import Path

# 依你的實際位置設定：若在子資料夾 App/ 裡，就改成 Path(__file__).parent/'App'/'app.py'
APP = Path(__file__).parent / "DCI strategy.py"     # 主程式檔名建議用全小寫 app.py
PORT = 8501

def main():
    if not APP.exists():
        print(f"[ERROR] app not found: {APP.resolve()}")
        return

    cmd = [
        sys.executable, "-m", "streamlit", "run", str(APP),
        "--server.address", "0.0.0.0",
        "--server.port", str(PORT),
        "--server.headless", "true",
    ]
    print("RUN:", " ".join(cmd))
    # 用 Popen 不中斷，或想看輸出就用 run
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
