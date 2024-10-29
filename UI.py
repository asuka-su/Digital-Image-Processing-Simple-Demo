from __future__ import annotations

import argparse
import pathlib
import gradio as gr
import numpy as np


def create_demo_hw1(process):
    with gr.Blocks() as demo:
        gr.Markdown('## Image Editing Based on HSL Space') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
                h_slider = gr.Slider(minimum=-1, maximum=1, label="Hue", value=0, step=0.01, visible=False)
                s_slider = gr.Slider(minimum=-1, maximum=1, label="Saturation", value=0, step=0.01, visible=False)
                l_slider = gr.Slider(minimum=-1, maximum=1, label="Lightness", value=0, step=0.01, visible=False)
            with gr.Column():
                with gr.Row():
                    output_image = gr.Image(type='numpy', label='output image', interactive=False, visible=True)
                with gr.Row():
                    with gr.Column(min_width=100):
                        r_channel = gr.Image(type='numpy', label = 'R channel', interactive=False)
                        r_box = gr.Checkbox(value=True, label='visible', interactive=True, visible=False)
                    with gr.Column(min_width=100):
                        g_channel = gr.Image(type='numpy', label = 'G channel', interactive=False)
                        g_box = gr.Checkbox(value=True, label='visible', interactive=True, visible=False)
                    with gr.Column(min_width=100):
                        b_channel = gr.Image(type='numpy', label = 'B channel', interactive=False)
                        b_box = gr.Checkbox(value=True, label='visible', interactive=True, visible=False)

        def input_update(input_image):
            if input_image is not None:
                return (
                    *process(input_image, h_slider.value, s_slider.value, l_slider.value, r_box.value, g_box.value, b_box.value), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                )
            return (
                None, None, None, None, 
                gr.update(value=0, visible=False), 
                gr.update(value=0, visible=False), 
                gr.update(value=0, visible=False), 
                gr.update(value=True, visible=False), 
                gr.update(value=True, visible=False), 
                gr.update(value=True, visible=False), 
            )

        input_image.change(fn=input_update, inputs=[input_image], outputs=[output_image, r_channel, g_channel, b_channel, h_slider, s_slider, l_slider, r_box, g_box, b_box])

        def slider_update(slider):
            return slider.release(
                fn=process, 
                inputs=[input_image, h_slider, s_slider, l_slider, r_box, g_box, b_box], 
                outputs=[output_image, r_channel, g_channel, b_channel]
            )
        
        slider_update(h_slider)
        slider_update(s_slider)
        slider_update(l_slider)

        def checkbox_update(checkbox):
            return checkbox.input(
                fn=process, 
                inputs=[input_image, h_slider, s_slider, l_slider, r_box, g_box, b_box], 
                outputs=[output_image, r_channel, g_channel, b_channel]
            )
        
        checkbox_update(r_box)
        checkbox_update(g_box)
        checkbox_update(b_box)

    return demo


def create_demo_hw2(process):
    with gr.Blocks() as demo:
        gr.Markdown('## Image Resize') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
                method = gr.Radio(choices=['Nearest', 'Bilinear', 'Bicubic', 'Lanczos'], label='Resize Method', interactive=True)
                scaling = gr.Radio(choices=['x0.5', 'x1', 'x2', 'x4'], label='Scale Factor', interactive=True)
                cv2_box = gr.Checkbox(value=False, label='Use opencv-python', interactive=True)
            with gr.Column():
                output_image = gr.Image(type='numpy', label='output image', interactive=False)
                run_button = gr.Button(value='START!')       
        
        input_image.change(fn=lambda: None, inputs=[], outputs=[output_image])

        run_button.click(fn=process,
                        inputs=[input_image, method, scaling, cv2_box],
                        outputs=[output_image])
    return demo


def create_demo_hw2_ex(process):
    with gr.Blocks() as demo:
        gr.Markdown('## Image Rotation and Shearing') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
                method = gr.Radio(choices=['Rotation', 'Shearing'], label='Basic Transforms')
                with gr.Row():
                    center_x = gr.Textbox(label='Center x', visible=False)
                    center_y = gr.Textbox(label='Center y', visible=False)
                    deg = gr.Slider(minimum=0, maximum=360, label='Degree', value=0, step=1, visible=False)
                with gr.Row():
                    side = gr.Radio(choices=['Left', 'Right', 'Up', 'Down'], label='Shearing Side', visible=False, scale=1)
                    size = gr.Slider(minimum=-1, maximum=1, label='Shearing Size', value=0, step=0.1, visible=False, scale=2)
            with gr.Column():
                output_image = gr.Image(type='numpy', label='output image', interactive=False)
                run_button = gr.Button(value='START!')       
                with gr.Row():
                    with gr.Column(scale=2, min_width=100):
                        R_value = gr.Textbox(label='R value', value='0', interactive=False)
                        G_value = gr.Textbox(label='G value', value='0', interactive=False)
                        B_value = gr.Textbox(label='B value', value='0', interactive=False)
                    with gr.Column(scale=1, min_width=40):
                        quick_setting = gr.Radio(choices=['Black', 'White', 'Other'], label='Default Color Quick Setting', value='Black')

        input_image.change(fn=lambda: None, inputs=[], outputs=output_image)

        def method_radio_update(radio):
            if radio == 'Rotation':
                return (
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=True), 
                    gr.update(visible=False), 
                    gr.update(visible=False)
                )
            elif radio == 'Shearing':
                return (
                    gr.update(visible=False), 
                    gr.update(visible=False), 
                    gr.update(visible=False), 
                    gr.update(visible=True), 
                    gr.update(visible=True)
                )
            else:
                raise gr.Error(f"Unknown operation choice {radio}", duration=5)

        method.change(fn=method_radio_update, inputs=[method], outputs=[center_x, center_y, deg, side, size])

        def setting_radio_update(radio):
            if radio == 'Black':
                return (
                    gr.update(value=0, interactive=False), 
                    gr.update(value=0, interactive=False), 
                    gr.update(value=0, interactive=False)
                )
            elif radio == 'White':
                return (
                    gr.update(value=255, interactive=False), 
                    gr.update(value=255, interactive=False), 
                    gr.update(value=255, interactive=False), 
                )
            elif radio == 'Other':
                return (
                    gr.update(interactive=True), 
                    gr.update(interactive=True), 
                    gr.update(interactive=True), 
                )
            else:
                raise gr.Error(f"Unknown default color value choice {radio}", duration=5)
        
        quick_setting.change(fn=setting_radio_update, inputs=[quick_setting], outputs=[R_value, G_value, B_value])

        run_button.click(fn=process,
                        inputs=[input_image, method, center_x, center_y, deg, R_value, G_value, B_value, side, size],
                        outputs=[output_image])
    return demo


def create_demo_hw3(process):
    with gr.Blocks() as demo:
        gr.Markdown('## NOT IMPLEMENTED') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='output image', interactive=False)
                run_button = gr.Button(value='START!')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw4(process):
    with gr.Blocks() as demo:
        gr.Markdown('## NOT IMPLEMENTED') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='output image', interactive=False)
                run_button = gr.Button(value='START!')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo

def create_demo_hw5(process):
    with gr.Blocks() as demo:
        gr.Markdown('## NOT IMPLEMENTED') 
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(sources=['upload', 'webcam', 'clipboard'], type='numpy', label='input image')  
            with gr.Column():
                output_image = gr.Image(type='numpy', label='output image', interactive=False)
                run_button = gr.Button(value='START!')

        run_button.click(fn=process,
                        inputs=[input_image],
                        outputs=[output_image])
    return demo