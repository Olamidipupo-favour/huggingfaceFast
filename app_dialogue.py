import os
import subprocess

# Install flash attention
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)


import copy
import spaces
import time
import torch

from threading import Thread
from typing import List, Dict, Union
import urllib
from urllib.parse import urlparse
from PIL import Image
import io
import pandas as pd
import datasets
import json
import requests

import gradio as gr
from transformers import AutoProcessor, TextIteratorStreamer
from transformers import Idefics2ForConditionalGeneration


DEVICE = torch.device("cuda")
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=os.environ["HF_AUTH_TOKEN"],
    ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    token=os.environ["HF_AUTH_TOKEN"],
)

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called Assistant, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about the content of visual inputs. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.",
            },
        ],
    }
]

API_TOKEN = os.getenv("HF_AUTH_TOKEN")
HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")
# IDEFICS_LOGO = "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/IDEFICS_logo.png"
BOT_AVATAR = "IDEFICS_logo.png"


# Chatbot utils
def turn_is_pure_media(turn):
    return turn[1] is None


def load_image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image_data = response.read()
        image_stream = io.BytesIO(image_data)
        image = Image.open(image_stream)
        return image


def img_to_bytes(image_path):
    image = Image.open(image_path)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_bytes = buffer.getvalue()
    image.close()
    return img_bytes


def format_user_prompt_with_im_history_and_system_conditioning(
    user_prompt, chat_history
) -> List[Dict[str, Union[List, str]]]:
    """
    Produces the resulting list that needs to go inside the processor.
    It handles the potential image(s), the history and the system conditionning.
    """
    resulting_messages = copy.deepcopy(SYSTEM_PROMPT)
    resulting_images = []
    for resulting_message in resulting_messages:
        if resulting_message["role"] == "user":
            for content in resulting_message["content"]:
                if content["type"] == "image":
                    resulting_images.append(load_image_from_url(content["image"]))

    # Format history
    for turn in chat_history:
        if not resulting_messages or (
            resulting_messages and resulting_messages[-1]["role"] != "user"
        ):
            resulting_messages.append(
                {
                    "role": "user",
                    "content": [],
                }
            )

        if turn_is_pure_media(turn):
            media = turn[0][0]
            resulting_messages[-1]["content"].append({"type": "image"})
            resulting_images.append(Image.open(media))
        else:
            user_utterance, assistant_utterance = turn
            resulting_messages[-1]["content"].append(
                {"type": "text", "text": user_utterance.strip()}
            )
            resulting_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": user_utterance.strip()}],
                }
            )

    # Format current input
    if not user_prompt["files"]:
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt["text"]}],
            }
        )
    else:
        # Choosing to put the image first (i.e. before the text), but this is an arbiratrary choice.
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}] * len(user_prompt["files"])
                + [{"type": "text", "text": user_prompt["text"]}],
            }
        )
        resulting_images.extend([Image.open(im["path"]) for im in user_prompt["files"]])

    return resulting_messages, resulting_images


def extract_images_from_msg_list(msg_list):
    all_images = []
    for msg in msg_list:
        for c_ in msg["content"]:
            if isinstance(c_, Image.Image):
                all_images.append(c_)
    return all_images


@spaces.GPU(duration=180)
def model_inference(
    user_prompt,
    chat_history,
    model_selector,
    decoding_strategy,
    temperature,
    max_new_tokens,
    repetition_penalty,
    top_p,
):
    if user_prompt["text"].strip() == "" and not user_prompt["files"]:
        gr.Error("Please input a query and optionally image(s).")

    if user_prompt["text"].strip() == "" and user_prompt["files"]:
        gr.Error("Please input a text query along the image(s).")

    for file in user_prompt["files"]:
        if not file["mime_type"].startswith("image/"):
            gr.Error("Idefics2 only supports images. Please input a valid image.")

    streamer = TextIteratorStreamer(
        PROCESSOR.tokenizer,
        skip_prompt=True,
        timeout=5.0,
    )

    # Common parameters to all decoding strategies
    # This documentation is useful to read: https://huggingface.co/docs/transformers/main/en/generation_strategies
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    assert decoding_strategy in [
        "Greedy",
        "Top P Sampling",
    ]
    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    # Creating model inputs
    (
        resulting_text,
        resulting_images,
    ) = format_user_prompt_with_im_history_and_system_conditioning(
        user_prompt=user_prompt,
        chat_history=chat_history,
    )
    prompt = PROCESSOR.apply_chat_template(resulting_text, add_generation_prompt=True)
    inputs = PROCESSOR(
        text=prompt,
        images=resulting_images if resulting_images else None,
        return_tensors="pt",
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generation_args.update(inputs)

    # # The regular non streaming generation mode
    # _ = generation_args.pop("streamer")
    # generated_ids = MODELS[model_selector].generate(**generation_args)
    # generated_text = PROCESSOR.batch_decode(generated_ids[:, generation_args["input_ids"].size(-1): ], skip_special_tokens=True)[0]
    # return generated_text

    # The streaming generation mode
    thread = Thread(
        target=MODELS[model_selector].generate,
        kwargs=generation_args,
    )
    thread.start()

    print("Start generating")
    acc_text = ""
    for text_token in streamer:
        time.sleep(0.04)
        acc_text += text_token
        if acc_text.endswith("<end_of_utterance>"):
            acc_text = acc_text[:-18]
        yield acc_text
    print("Success - generated the following text:", acc_text)
    print("-----")


def flag_chat(
    model_selector,
    chat_history,
    decoding_strategy,
    temperature,
    max_new_tokens,
    repetition_penalty,
    top_p,
):
    images = []
    text_flag = []
    prev_ex_is_image = False
    for ex in chat_history:
        if isinstance(ex[0], dict):
            images.append(ex[0]["file"])
            prev_ex_is_image = True
        else:
            if prev_ex_is_image:
                text_flag.append([f"User:<image>{ex[0]}", f"Assistant:{ex[1]}"])
            else:
                text_flag.append([f"User:{ex[0]}", f"Assistant:{ex[1]}"])
            prev_ex_is_image = False
    image_flag = images[0]
    dope_dataset_writer.flag(
        flag_data=[
            model_selector,
            images[0],
            text_flag,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ]
    )


# Hyper-parameters for generation
max_new_tokens = gr.Slider(
    minimum=8,
    maximum=1024,
    value=512,
    step=1,
    interactive=True,
    label="Maximum number of new tokens to generate",
)
repetition_penalty = gr.Slider(
    minimum=0.01,
    maximum=5.0,
    value=1.1,
    step=0.01,
    interactive=True,
    label="Repetition penalty",
    info="1.0 is equivalent to no penalty",
)
decoding_strategy = gr.Radio(
    [
        "Greedy",
        "Top P Sampling",
    ],
    value="Greedy",
    label="Decoding strategy",
    interactive=True,
    info="Higher values is equivalent to sampling more low-probability tokens.",
)
temperature = gr.Slider(
    minimum=0.0,
    maximum=5.0,
    value=0.4,
    step=0.1,
    interactive=True,
    label="Sampling temperature",
    info="Higher values will produce more diverse outputs.",
)
top_p = gr.Slider(
    minimum=0.01,
    maximum=0.99,
    value=0.8,
    step=0.01,
    interactive=True,
    label="Top P",
    info="Higher values is equivalent to sampling more low-probability tokens.",
)


chatbot = gr.Chatbot(
    label="Idefics2",
    avatar_images=[None, BOT_AVATAR],
    height=450,
)

dope_dataset_writer = gr.HuggingFaceDatasetSaver(
    HF_WRITE_TOKEN, "HuggingFaceM4/dope-dataset", private=True
)
problematic_dataset_writer = gr.HuggingFaceDatasetSaver(
    HF_WRITE_TOKEN, "HuggingFaceM4/problematic-dataset", private=True
)
# Using Flagging for saving dope and problematic examples
# Dope examples flagging


# gr.Markdown("""## How to use?

#     There are two ways to provide image inputs:
#     - Using the image box on the left panel
#     - Using the inline syntax: `text<fake_token_around_image><image:URL_IMAGE><fake_token_around_image>text`

#     The second syntax allows inputting an arbitrary number of images.""")

image_flag = gr.Image(visible=False)
text_flag = gr.Textbox(visible=False)
with gr.Blocks(
    fill_height=True,
    css=""".gradio-container .avatar-container {height: 40px width: 40px !important;}""",
) as demo:
    # model selector should be set to `visbile=False` ultimately
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=MODELS.keys(),
            value=list(MODELS.keys())[0],
            interactive=True,
            show_label=False,
            container=False,
            label="Model",
            visible=True,
        )

    decoding_strategy.change(
        fn=lambda selection: gr.Slider(
            visible=(
                selection
                in [
                    "contrastive_sampling",
                    "beam_sampling",
                    "Top P Sampling",
                    "sampling_top_k",
                ]
            )
        ),
        inputs=decoding_strategy,
        outputs=temperature,
    )
    decoding_strategy.change(
        fn=lambda selection: gr.Slider(
            visible=(
                selection
                in [
                    "contrastive_sampling",
                    "beam_sampling",
                    "Top P Sampling",
                    "sampling_top_k",
                ]
            )
        ),
        inputs=decoding_strategy,
        outputs=repetition_penalty,
    )
    decoding_strategy.change(
        fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
        inputs=decoding_strategy,
        outputs=top_p,
    )

    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        # examples=[{"text": "hello"}, {"text": "hola"}, {"text": "merhaba"}],
        title="Idefics2 Playground",
        multimodal=True,
        additional_inputs=[
            model_selector,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
    )
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1, min_width=50):
                dope_bttn = gr.Button("Dope🔥")
            with gr.Column(scale=1, min_width=50):
                problematic_bttn = gr.Button("Problematic😬")

    dope_dataset_writer.setup(
        [
            model_selector,
            image_flag,
            text_flag,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        "gradio_dope_data_points",
    )
    dope_bttn.click(
        fn=flag_chat,
        inputs=[
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=None,
        preprocess=False,
    )
    # Problematic examples flagging
    problematic_dataset_writer.setup(
        [
            model_selector,
            image_flag,
            text_flag,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        "gradio_problematic_data_points",
    )
    problematic_bttn.click(
        fn=flag_chat,
        inputs=[
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        outputs=None,
        preprocess=False,
    )

demo.launch()
