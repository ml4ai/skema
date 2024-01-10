# -*- coding: utf-8 -*-
"""
ENV-based config
"""

import os


SKEMA_RS_DEFAULT_TIMEOUT = float(os.environ.get("SKEMA_RS_DEFAULT_TIMEOUT", "60.0"))