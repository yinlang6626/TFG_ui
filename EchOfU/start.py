#!/usr/bin/env python
import os
import sys
import webbrowser
import threading
import time
from app import app

def open_browser():
    """等待服务器启动后自动打开浏览器"""
    time.sleep(2)  # 等待Flask启动
    webbrowser.open('http://127.0.0.1:5001')

if __name__ == '__main__':
    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动Flask
    app.run(debug=True, port=5001)