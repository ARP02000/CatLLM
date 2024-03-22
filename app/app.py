import random
import gradio
import gradio as gr
import time
import json
import wave
from zhipuai import ZhipuAI
import os
from presets import *
from utils import *
import shutil
import numpy as np
import ffmpeg
import socket
import datetime

api_key="ab0140939aef9898d0ce3939963ff0d9.c8o20urtFSfkp5FJ"
client = ZhipuAI(api_key=api_key)

# 加载模型
model_path = "./models/whisper-finetune-ct2"
use_gpu = True
use_int8 = True
beam_size = 10
language = "zh"
vad_filter = True

if use_gpu:
    if not use_int8:
        model = WhisperModel(model_path, device="cuda", compute_type="float16", num_workers=1,
                             local_files_only=True)
    else:
        model = WhisperModel(model_path, device="cuda", compute_type="int8_float16", num_workers=1,
                             local_files_only=True)
else:
    model = WhisperModel(model_path, device="cpu", compute_type="int8", num_workers=1,
                         local_files_only=True)


def answer(history_record, chatbot):
    """
    :param history_record:
    :param chatbot:
    :return:
    """
    print("------------answer---------------------")
    print(chatbot, type(chatbot))
    print("answer")
    print(history_record, chatbot)
    if history_record[-1]['content'] != "":
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=history_record,
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                chatbot[-1][1] += chunk.choices[0].delta.content
        history_record.append({"role":"assistant", "content":chatbot[-1][1]})
        print("after history:", history_record)
        print("after chatbot:", chatbot)

    return chatbot


def ask(user_input, chatbot, history_record):
    """
    :param user_input:
    :param chatbot:
    :param history_record:
    :return:
    """
    history_record.append({"role": "user", "content": user_input})
    chatbot += [[user_input, ""]]

    print(history_record, chatbot)
    return "", history_record, chatbot


def retry(history_record, chatbot):
    """
    :param history_record:
    :param chatbot:
    :return:
    """
    logging.info("Retry...")
    if len(history_record) == 0:
        return [["Empty context", ""]]
    else:
        chatbot[-1][1] = ""
        history_record.pop()
        print("retry------------")
        print(history_record)
        chatbot = answer(history_record, chatbot)
        return chatbot


def transcribe(audio_path):
    # 语音识别
    segments, info = model.transcribe(audio_path, beam_size=beam_size, language=language,
                                      vad_filter=vad_filter)
    res = ""
    for segment in segments:
        res += segment.text
    return res


def save_wavfile(src_path, file_id):
    hostname = socket.gethostname()
    dst_path = f"./audio_file/{hostname}/{datetime.datetime.now().strftime('%Y-%m-%d')}"
    if os.path.exists(dst_path):
        pass
    else:
        os.mkdir(dst_path)

    new_filename = os.path.join(dst_path, f"audio_{file_id}.wav")
    if os.path.exists(new_filename):
        os.remove(new_filename)
    ffmpeg.input(src_path).output(new_filename, ar=16000).run()

    file_id += 1
    return new_filename, file_id


with gr.Blocks() as demo:
    # 创建一个chatbot和消息发送框
    history_record = gr.State([])
    file_id = gr.State(0)

    with gr.Row():
        gr.HTML(title)
        status_display = gr.Markdown("Success", elem_id="status_display")
    gr.Markdown(description_top)
    chatbot = gr.Chatbot(
        [],
        avatar_images=(None, (r"./pic/猫猫头.jpg"))
    )

    with gr.Row():
        user_input = gr.Textbox(
            scale=4,
            show_label=False,
            container=False,
            lines=2
        )

        submitBtn = gr.Button("Send")
        cancelBtn = gr.Button("Stop")
    with gr.Row():
        audio = gr.Audio(
            sources=['microphone'],
            type='filepath',
            min_length=1,
            max_length=15
        )

    with gr.Row():
        clear = gr.ClearButton([user_input, chatbot])
        delLastBtn = gr.Button("🗑️ Remove Last Turn")
        retryBtn = gr.Button("🔄 Regenerate")

    gr.Markdown(description)

    save_wavfile_args = dict(
        fn=save_wavfile,
        inputs=[
            audio,
            file_id
        ],
        outputs=[
            audio,
            file_id
        ],
        show_progress = True,
    )

    transcribe_args = dict(
        fn=transcribe,
        inputs=[audio],
        outputs=[user_input],
        show_progress=True,
    )

    answer_args = dict(
        fn=answer,
        inputs=[
            history_record,
            chatbot,
        ],
        outputs=[chatbot],
        show_progress=True,
    )

    ask_args = dict(
        fn=ask,
        inputs=[
            user_input,
            chatbot,
            history_record
        ],
        outputs=[user_input, history_record, chatbot],
        show_progress=True,
    )

    transfer_input_args = dict(
        fn=transfer_input,
        inputs=[user_input],
        outputs=[user_input, user_input, submitBtn],
        show_progress=True
    )

    predict_event1 = user_input.submit(**ask_args).then(**answer_args)

    predict_event2 = submitBtn.click(
        fn=ask,
        inputs=[
            user_input,
            chatbot,
            history_record
        ],
        outputs=[user_input, history_record, chatbot],
        show_progress=True,
        cancels=predict_event1
    ).then(**answer_args)

    predict_event3 = retryBtn.click(
        fn=retry,
        inputs=[history_record, chatbot],
        outputs=[chatbot],
        cancels=[predict_event1, predict_event2],
    )

    delLastBtn.click(
        fn=delete_last_conversation,
        inputs=[chatbot, history_record],
        outputs=[chatbot, history_record, status_display],
        show_progress=True,
        cancels=[predict_event1, predict_event2, predict_event3]
    )

    predict_event4 = audio.stop_recording(
        fn=transcribe,
        inputs=[audio],
        outputs=[user_input]
    ).then(**save_wavfile_args).then(**transcribe_args)

    predict_event5 = audio.upload(
        fn=save_wavfile,
        inputs=[audio, file_id],
        outputs=[audio, file_id]
    ).then(**transcribe_args)

    cancelBtn.click(
        fn=cancel_outputing,
        inputs=[],
        outputs=[status_display],
        cancels=[predict_event1, predict_event2, predict_event3]
    )

demo.queue()
demo.launch(share=True)
