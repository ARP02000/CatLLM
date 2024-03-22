from __future__ import annotations
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Type
import logging
import json
import os
import datetime
import hashlib
import csv
import requests
import re
import html
import sys
import gc
from pygments.lexers import guess_lexer, ClassNotFound
import gradio as gr
from pygments import highlight
from pygments.lexers import guess_lexer,get_lexer_by_name
from pygments.formatters import HtmlFormatter
from presets import *
import os
from faster_whisper import WhisperModel


def reset_textbox():
    return gr.update(value=""), ""


def transfer_input(inputs):
    # 一次性返回，降低延迟
    textbox = reset_textbox()
    return (
        inputs,
        gr.update(value=""),
        gr.Button(visible=True),
    )


def reset_state():
    return [], [], "Reset Done"


def delete_last_conversation(chatbot, history):
    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0:
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )


def cancel_outputing():
    return "Stop Done"

