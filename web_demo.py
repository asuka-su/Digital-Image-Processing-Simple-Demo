from __future__ import annotations

import argparse
import pathlib
import gradio as gr
from UI import *
from functions import *


HTML_DESCRIPTION = '''
<div align=center>
<h1 style="font-weight: 900; margin-bottom: 7px;">
   图像处理网页演示工具</a>
</h1>
<p>作者: asuka su</p>
</div>
'''

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(HTML_DESCRIPTION) 
    with gr.Tabs():
        with gr.TabItem('RGB & HSL'):
            create_demo_hw1(function_hw1)          
        with gr.TabItem('RESIZE'):
            create_demo_hw2(function_hw2)   
        with gr.TabItem('ROTATION & SHEARING'):
            create_demo_hw2_ex(function_hw2_ex)
        with gr.TabItem('NOT IMPLEMENTED'):
            create_demo_hw3(function_hw3)  
        with gr.TabItem('NOT IMPLEMENTED'):
            create_demo_hw4(function_hw4) 
        with gr.TabItem('NOT IMPLEMENTED'):
            create_demo_hw5(function_hw5)           

if __name__ == '__main__':
    demo.launch(server_port=8088, share=True)