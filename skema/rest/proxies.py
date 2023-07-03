# -*- coding: utf-8 -*-
"""
Service proxies.
"""

import os

# MORAE etc
SKEMA_RS_ADDESS = os.environ.get("SKEMA_RS_ADDRESS", "https://skema-rs.askem.lum.ai")


# MathJAX service
SKEMA_MATHJAX_PROTOCOL = os.environ.get('SKEMA_MATHJAX_PROTOCOL', 'http://')
SKEMA_MATHJAX_HOST = os.environ.get('SKEMA_MATHJAX_HOST', '127.0.0.1')
SKEMA_MATHJAX_PORT = str(os.environ.get('SKEMA_MATHJAX_PORT', 8031))
SKEMA_MATHJAX_ADDRESS = os.environ.get("SKEMA_MATHJAX_ADDRESS", f"{SKEMA_MATHJAX_PROTOCOL}{SKEMA_MATHJAX_HOST}:{SKEMA_MATHJAX_PORT}")