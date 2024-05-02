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
from PIL import Image
import io
import datasets

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
                "text": "The following is a conversation between Idefics2, a highly knowledgeable and intelligent visual AI assistant created by Hugging Face, referred to as Assistant, and a human user called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about them, but it cannot generate images. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.",
            },
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello, I'm Idefics2, Huggingface's latest multimodal assistant. How can I help you?",
            },
        ],
    }
]
examples_path = os.path.dirname(__file__)
EXAMPLES = [
    [
        {
            "text": "For 2024, the interest expense is twice what it was in 2014, and the long-term debt is 10% higher than its 2015 level. Can you calculate the combined total of the interest and long-term debt for 2024?",
            "files": [f"{examples_path}/example_images/mmmu_example_2.png"],
        }
    ],
    [
        {
            "text": "What's in the image?",
            "files": [f"{examples_path}/example_images/plant_bulb.webp"],
        }
    ],
    [
        {
            "text": "Describe the image",
            "files": [f"{examples_path}/example_images/baguettes_guarding_paris.png"],
        }
    ],
    [
        {
            "text": "Read what's written on the paper",
            "files": [f"{examples_path}/example_images/paper_with_text.png"],
        }
    ],
    [
        {
            "text": "The respective main characters of these two movies meet in real life. Imagine their discussion. It should be sassy, and the beginning of a mysterious adventure.",
            "files": [f"{examples_path}/example_images/barbie.jpeg", f"{examples_path}/example_images/oppenheimer.jpeg"],
        }
    ],
    [
        {
            "text": "Can you explain this meme?",
            "files": [f"{examples_path}/example_images/running_girl_meme.webp"],
        }
    ],
    [
        {
            "text": "What happens to fish if pelicans increase?",
            "files": [f"{examples_path}/example_images/ai2d_example_2.jpeg"],
        }
    ],
    [
        {
            "text": "Give an art-critic description of this well known painting",
            "files": [f"{examples_path}/example_images/Van-Gogh-Starry-Night.jpg"],
        }
    ],
    [
        {
            "text": "Chase wants to buy 4 kilograms of oval beads and 5 kilograms of star-shaped beads. How much will he spend?",
            "files": [f"{examples_path}/example_images/mmmu_example.jpeg"],
        }
    ],
    [
        {
            "text": "Write an online ad for that product.",
            "files": [f"{examples_path}/example_images/shampoo.jpg"],
        }
    ],
    [
        {
            "text": "Describe this image in detail and explain why it is disturbing.",
            "files": [f"{examples_path}/example_images/cat_cloud.jpeg"],
        }
    ],
    [
        {
            "text": "Why is this image cute?",
            "files": [
                f"{examples_path}/example_images/kittens-cats-pet-cute-preview.jpg"
            ],
        }
    ],
    [
        {
            "text": "What is formed by the deposition of either the weathered remains of other rocks?",
            "files": [f"{examples_path}/example_images/ai2d_example.jpeg"],
        }
    ],    
    [
        {
            "text": "What's funny about this image?",
            "files": [f"{examples_path}/example_images/pope_doudoune.webp"],
        }
    ],
    [
        {
            "text": "Can this happen in real life?",
            "files": [f"{examples_path}/example_images/elephant_spider_web.webp"],
        }
    ],
    [
        {
            "text": "What's unusual about this image?",
            "files": [f"{examples_path}/example_images/dragons_playing.png"],
        }
    ],
    [
        {
            "text": "Why is that image comical?",
            "files": [f"{examples_path}/example_images/eye_glasses.jpeg"],
        }
    ],
]

API_TOKEN = os.getenv("HF_AUTH_TOKEN")
HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")
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
    image = Image.open(image_path).convert(mode='RGB')
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
        resulting_images.extend([Image.open(path) for path in user_prompt["files"]])

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


FEATURES = datasets.Features(
    {
        "model_selector": datasets.Value("string"),
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "conversation": datasets.Sequence({"User": datasets.Value("string"), "Assistant": datasets.Value("string")}),
        "decoding_strategy": datasets.Value("string"),
        "temperature": datasets.Value("float32"),
        "max_new_tokens": datasets.Value("int32"),
        "repetition_penalty": datasets.Value("float32"),
        "top_p": datasets.Value("int32"),
        }
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
    visible=False,
    interactive=True,
    label="Sampling temperature",
    info="Higher values will produce more diverse outputs.",
)
top_p = gr.Slider(
    minimum=0.01,
    maximum=0.99,
    value=0.8,
    step=0.01,
    visible=False,
    interactive=True,
    label="Top P",
    info="Higher values is equivalent to sampling more low-probability tokens.",
)


chatbot = gr.Chatbot(
    label="Idefics2-Chatty",
    avatar_images=[None, BOT_AVATAR],
    height=450,
)

with gr.Blocks(
    fill_height=True,
    css=""".gradio-container .avatar-container {height: 40px width: 40px !important;} #duplicate-button {margin: auto; color: white; background: #f1a139; border-radius: 100vh;}""",
) as demo:

    gr.Markdown("# 🐶 Hugging Face Idefics2 8B Chatty")
    gr.Markdown("In this demo you'll be able to chat with [Idefics2-8B-chatty](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty), a variant of [Idefics2-8B](https://huggingface.co/HuggingFaceM4/idefics2-8b-chatty) further fine-tuned on chat datasets.")
    gr.Markdown("If you want to learn more about Idefics2 and its variants, you can check our [blog post](https://huggingface.co/blog/idefics2).")
    gr.DuplicateButton(value="Duplicate Space for private use", elem_id="duplicate-button")
    # model selector should be set to `visbile=False` ultimately
    with gr.Row(elem_id="model_selector_row"):
        model_selector = gr.Dropdown(
            choices=MODELS.keys(),
            value=list(MODELS.keys())[0],
            interactive=True,
            show_label=False,
            container=False,
            label="Model",
            visible=False,
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
        fn=lambda selection: gr.Slider(visible=(selection in ["Top P Sampling"])),
        inputs=decoding_strategy,
        outputs=top_p,
    )

    gr.ChatInterface(
        fn=model_inference,
        chatbot=chatbot,
        examples=EXAMPLES,
        multimodal=True,
        cache_examples=False,
        additional_inputs=[
            model_selector,
            decoding_strategy,
            temperature,
            max_new_tokens,
            repetition_penalty,
            top_p,
        ],
    )

demo.launch()
