import random
import gradio
import gradio as gr
import time
import json
import wave
from zhipuai import ZhipuAI
import copy
import os
from presets import *
from utils import *
import shutil
import numpy as np
import ffmpeg
import socket
import datetime
from inference import get_tts_wav, transcribe
from gradio_multimodalchatbot import MultimodalChatbot
from gradio.data_classes import FileData


api_key="your-apikey"
client = ZhipuAI(api_key=api_key)



def answer(user_input, history_record, chatbot):
    """
    :param history_record:
    :param chatbot:
    :return:
    """
    user_input = user_input.strip()
    if len(user_input) == 0:
        return "", history_record, chatbot
    else:
        history_record.append({"role": "user", "content": user_input})
        user_input_dict = {
            "text": user_input,
            "files": []
        }
        chatbot += [[user_input_dict, {"text": "", "files": []}]]

        print("------------answer---------------------")
        print(chatbot, type(chatbot))
        print("answer")
        print(history_record, chatbot)

        prompt_record = copy.deepcopy(history_record)
        prompt_record[-1]["content"] += " 回答不超过30个字"
        question = user_input.replace("\n", "").replace("-", "")
        if question != "":
            response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=prompt_record,
                stream=True,
            )

            message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    message += chunk.choices[0].delta.content
                    chatbot[-1][-1]["text"] += chunk.choices[0].delta.content
            history_record.append({"role": "assistant", "content": message})
            print("after history:", history_record)
            print("after chatbot:", chatbot)
            chatbot[-1][-1]["text"] = message

        filename = get_tts_wav(chatbot[-1][-1]["text"], question)
        chatbot[-1][-1]["files"] = [{"file": FileData(path=filename)}]
        return "", history_record, chatbot


def retry(history_record, chatbot):
    """
    :param history_record:
    :param chatbot:
    :return:
    """
    if len(history_record) == 0:
        pass
    else:
        history_record.pop()
        user_input = history_record[-1]["content"]
        history_record.pop()
        chatbot.pop()
        print("retry------------")
        print(history_record, user_input, chatbot)
        _, history_record, chatbot = answer(user_input, history_record, chatbot)
        return "", history_record, chatbot


def save_wavfile(src_path, file_id):
    hostname = socket.gethostname()
    dst_path = f"./audio/{hostname}/{datetime.datetime.now().strftime('%Y-%m-%d')}"
    if os.path.exists(dst_path):
        pass
    else:
        os.makedirs(dst_path)

    new_filename = os.path.join(dst_path, f"audio_{file_id}_{datetime.datetime.now().microsecond}.wav")
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
    chatbot = MultimodalChatbot(
        [],
        avatar_images=(None, "./pic/猫猫头.jpg"),
        show_copy_button=True,
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
            user_input,
            history_record,
            chatbot,
        ],
        outputs=[user_input, history_record, chatbot],
        show_progress=True,
    )

    predict_event1 = user_input.submit(**answer_args)

    predict_event2 = submitBtn.click(
        fn=answer,
        inputs=[
            user_input,
            history_record,
            chatbot
        ],
        outputs=[user_input, history_record, chatbot],
        cancels=predict_event1
    )

    predict_event3 = retryBtn.click(
        fn=retry,
        inputs=[history_record, chatbot],
        outputs=[user_input, history_record, chatbot],
        cancels=[predict_event1, predict_event2],
    )

    delLastBtn.click(
        fn=delete_last_conversation,
        inputs=[chatbot, history_record],
        outputs=[chatbot, history_record, status_display],
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
