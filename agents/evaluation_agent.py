"""
Evaluation Agent - ukládání vizualizačních skriptů a hodnocení, výpočet skóre
"""
import base64
import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from PIL import Image
from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

