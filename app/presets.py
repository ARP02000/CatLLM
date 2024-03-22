import gradio as gr

title = """<h1 align="left" style="min-width:200px; margin-top:0;"> <img src="https://weepcat.github.io/images/猫猫.png" width="32px" style="display: inline"> Chat with CatLLM </h1>"""
description_top = \
"""
<div align="left">
<p> Currently Running: <a href="https://zhipuai.cn/">ChatGLM</a></p>
</div>
"""

# <p>
# Disclaimer: The LLaMA model is a third-party version available on Hugging Face model hub. This demo should be used for research purposes only. Commercial use is strictly prohibited. The model output is not censored and the authors do not endorse the opinions in the generated content. Use at your own risk.
# </p >

description = """\
<div align="center" style="margin:16px 0">
The demo is built on <a href="https://github.com/WeepCat/CatLLM">CatLLM</a>.
</div>
"""

CONCURRENT_COUNT = 100
