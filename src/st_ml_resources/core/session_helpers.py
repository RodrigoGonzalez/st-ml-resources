import os
import platform
import sys

import psutil
import streamlit as st
import shutil

system_path = "/"

# Get disk usage statistics
total, used, free = shutil.disk_usage(system_path)

PLATFORM_INFO = {
    "platform_system": platform.system(),
    "platform_release": platform.release(),
    "platform_version": platform.version(),
    "platform_machine": platform.machine(),
    "platform_processor": platform.processor(),
    "cpu_count": os.cpu_count(),
    "cpu_percent": psutil.cpu_percent(),
    "use_dev_snowflake": platform.system() != "Darwin",
    "sys_modules": sys.modules,
    "platform_info_set": True,
    "memory_info": {
        "total": psutil.virtual_memory().total / (1024 ** 2),
        "available": psutil.virtual_memory().available / (1024 ** 2),
        "used": psutil.virtual_memory().used / (1024 ** 2),
        "percent": psutil.virtual_memory().percent,
    },
    "disk_info": {
        "total": total / (2 ** 30),
        "used": used / (2 ** 30),
        "free": free / (2 ** 30),
        "percent": (used / total) * 100,
    },
}


def add_defaults_to_session(session_dict):
    for k, v in session_dict.items():
        if k not in st.session_state:
            st.session_state[k] = v
