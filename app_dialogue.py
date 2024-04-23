import os
import subprocess

# Install flash attention
subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)


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
    # "idefics2-8b (sft)": Idefics2ForConditionalGeneration.from_pretrained(
    #     "HuggingFaceM4/idefics2-8b",
    #     torch_dtype=torch.bfloat16,
    #     _attn_implementation="flash_attention_2",
    #     trust_remote_code=True,
    #     token=os.environ["HF_AUTH_TOKEN"],
    # ).to(DEVICE),
    "idefics2-8b-chatty (chat-600)": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=os.environ["HF_AUTH_TOKEN"],
        revision="bb460e58294bcb02430df9fd126b3c522f867d83"
    ).to(DEVICE),
    "idefics2-8b-chatty (chat-50)": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
        trust_remote_code=True,
        token=os.environ["HF_AUTH_TOKEN"],
        revision="1d57ffb705199370f7875667cc8f54abd09b2004"
    ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    token=os.environ["HF_AUTH_TOKEN"],
)

# SYSTEM_PROMPT = [ # Deactivating the system propmpt for now, but if I were to reactivate it, I would need to a/ transform turns into dict for applying the chat template, b/ manually overwrite the `default_template` to add the first line (that is not part of any turns), in particular for handling the bos_token.
# #     """The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called Assistant, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about the content of visual inputs. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.

# The conversation begins:""",
#     """\nUser:""",
#     "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/kittens-cats-pet-cute-preview.jpg?download=true",
#     "Describe this image.<end_of_utterance>",
#     """\nAssistant: Five kittens are standing together in the center of the photograph. From the left to right, there is one orange kitten, two kittens white and black stripes, and two brown kittens with an orange head. They are in bright green grass and it looks like they are moving forward.<end_of_utterance>""",
#     "\nUser:How about this image?",
#     "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/puppy.jpg?download=true",
#     "Can you describe it too?<end_of_utterance>",
#     """\nAssistant: A dog is lying on the floor, looking at the camera. It is looking directly at you. The dog has a white body and brown patches on its face and ears. Its eyes are dark. Its nose is black, and it has long, floppy ears, white paws, long fur, big eyes.<end_of_utterance>""",
#     "\nUser: What kind of breed is it?<end_of_utterance>",
#     """\nAssistant: It looks like a Cavalier King Charles Spaniel. These breeds are known for their gentle, affectionate nature. Their long ears and big, brown eyes give them a sweet, expressive face. Cavaliers are small dogs weighing between 13 and 20 pounds. They have a silky, medium-length coat that comes in various colors, including black and tan, tricolor, red and white, or black.<end_of_utterance>""",
#     "\nUser: What can you tell me about this breed of dogs?<end_of_utterance>",
#     """\nAssistant: One specific characteristic of the Cavalier King Charles Spaniel is its friendly and affectionate nature. This breed is known for being extremely sociable and forming strong bonds with their owners. They are often described as "velcro dogs" because they love to be close to their human companions, whether it's sitting on their laps, cuddling on the couch, or simply following them around the house.<end_of_utterance>""",
#     "\nUser: ghjkhjabnufs<end_of_utterance>",
#     """\nAssistant: That doesn’t seem to be a word. Could you ask me another way?<end_of_utterance>""",
#     "\nUser: Do you like Cavalier King Charles Spaniel?<end_of_utterance>",
#     """\nAssistant: I do not have personal opinions as I’m just a computer program. However, cavaliers are known for being excellent family pets due to their gentle and patient demeanor, making them great with children and other pets. Their affectionate nature also makes them ideal therapy dogs, as they can provide comfort and emotional support to those in need.<end_of_utterance>""",
#     "\nUser: How many dogs do you see in this image?",
#     "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/tennis_tsonga.jpg?download=true",
#     "<end_of_utterance>",
#     """\nAssistant: There is no dogs in this image. The picture shows a tennis player jumping to volley the ball.<end_of_utterance>""",
# ]

SYSTEM_PROMPT = [
    {
        "role": "user",
        "content": [
             {"type": "image", "image": "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/kittens-cats-pet-cute-preview.jpg?download=true"},
             {"type": "text", "text": "Describe this image."},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Five kittens are standing together in the center of the photograph. From the left to right, there is one orange kitten, two kittens white and black stripes, and two brown kittens with an orange head. They are in bright green grass and it looks like they are moving forward."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "How about this image?"},
            {"type": "image", "image": "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/puppy.jpg?download=true"},
            {"type": "text", "text": "Can you describe it too?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
                {"type": "text", "text": "A dog is lying on the floor, looking at the camera. It is looking directly at you. The dog has a white body and brown patches on its face and ears. Its eyes are dark. Its nose is black, and it has long, floppy ears, white paws, long fur, big eyes."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What can you tell me about this breed of dogs?"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "One specific characteristic of the Cavalier King Charles Spaniel is its friendly and affectionate nature. This breed is known for being extremely sociable and forming strong bonds with their owners. They are often described as \"velcro dogs\" because they love to be close to their human companions, whether it's sitting on their laps, cuddling on the couch, or simply following them around the house."},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "How many dogs do you see in the following image?"},
            {"type": "image", "image": "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/example_images/tennis_tsonga.jpg?download=true"},
        ],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "There are no dogs in this image. The picture shows a tennis player in the midst of a powerful swing."},
        ],
    },
]


API_TOKEN = os.getenv("HF_AUTH_TOKEN")
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
        if not resulting_messages or (resulting_messages and resulting_messages[-1]["role"] != "user"):
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
                    "content": [
                        {"type": "text", "text": user_utterance.strip()}
                    ]
                }
            )

    # Format current input
    if not user_prompt["files"]:
        resulting_messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt['text']}
                ],
            }
        )
    else:
        # Choosing to put the image first (i.e. before the text), but this is an arbiratrary choice.
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}] * len(user_prompt['files']) + [
                    {"type": "text", "text": user_prompt['text']}
                ]
            }
        )
        resulting_images.extend([Image.open(im['path']) for im in user_prompt['files']])

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
        timeout=5.,
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
    resulting_text, resulting_images = format_user_prompt_with_im_history_and_system_conditioning(
        user_prompt=user_prompt,
        chat_history=chat_history,
    )

    prompt = PROCESSOR.apply_chat_template(resulting_text, add_generation_prompt=True)
    inputs = PROCESSOR(text=prompt, images=resulting_images if resulting_images else None, return_tensors="pt")
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

dope_callback = gr.CSVLogger()
problematic_callback = gr.CSVLogger()


# Using Flagging for saving dope and problematic examples
    # Dope examples flagging
    

    # gr.Markdown("""## How to use?

    #     There are two ways to provide image inputs:
    #     - Using the image box on the left panel
    #     - Using the inline syntax: `text<fake_token_around_image><image:URL_IMAGE><fake_token_around_image>text`

    #     The second syntax allows inputting an arbitrary number of images.""")


with gr.Blocks(fill_height=True, css=""".gradio-container .avatar-container {height: 40px width: 40px !important;}""") as demo:
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
                selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
            )
        ),
        inputs=decoding_strategy,
        outputs=temperature,
    )
    decoding_strategy.change(
        fn=lambda selection: gr.Slider(
            visible=(
                selection in ["contrastive_sampling", "beam_sampling", "Top P Sampling", "sampling_top_k"]
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
        additional_inputs=[model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p],
    )
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1, min_width=50):
                dope_bttn = gr.Button("Dope🔥")
            with gr.Column(scale=1, min_width=50):
                problematic_bttn = gr.Button("Problematic😬")
    dope_callback.setup(
        [
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        "gradio_dope_data_points",
    )
    dope_bttn.click(
        lambda *args: dope_callback.flag(args),
        [
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        None,
        preprocess=False,
    )
    # Problematic examples flagging
    problematic_callback.setup(
        [
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        "gradio_problematic_data_points",
    )
    problematic_bttn.click(
        lambda *args: problematic_callback.flag(args),
        [
            model_selector,
            chatbot,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
        None,
        preprocess=False,
    )

demo.launch()
