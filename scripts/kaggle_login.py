"""
Kaggle Browser OAuth Login Launcher
===================================
Author: Sanish Gyawali
Launches Kaggle's OAuth login flow in your default browser and saves the token.
"""

import os
import sys

# Ensure unbuffered UTF-8 output
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

print("Starting Kaggle OAuth Login Flow...")
print("Opening your default browser to authorize...")

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.auth_login_cli(no_launch_browser=False)
except Exception as e:
    print(f"Error during Kaggle OAuth login: {e}")
    sys.exit(1)
