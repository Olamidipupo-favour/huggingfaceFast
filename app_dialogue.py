import copy
import os
import spaces
import subprocess
import torch

from threading import Thread
from typing import List, Tuple
from urllib.parse import urlparse
from PIL import Image

import gradio as gr
from gradio_client.client import DEFAULT_TEMP_DIR
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format

subprocess.run('pip install flash-attn --no-build-isolation', env={'FLASH_ATTENTION_SKIP_CUDA_BUILD': "TRUE"}, shell=True)

DEVICE = torch.device("cuda")
MODELS = {
    "282 - mix1 fixed - opt 23'000": AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/idefics2",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=os.environ["HF_AUTH_TOKEN"],
        revision="a1bc6a2b0f74cde25844144f602dde2808a564d9",
    ).to(DEVICE),
    "286 - mix6 tables - opt 20'000": AutoModelForCausalLM.from_pretrained(
        "HuggingFaceM4/idefics2",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        token=os.environ["HF_AUTH_TOKEN"],
        revision="b473d49caa964991b40b79fe7cb27d51d4d023f6",
    ).to(DEVICE),
    # "285 - continued pretraining on text sft - opt 2'000": AutoModelForCausalLM.from_pretrained(
    #     "HuggingFaceM4/idefics2",
    #     trust_remote_code=True,
    #     torch_dtype=torch.bfloat16,
    #     token=os.environ["HF_AUTH_TOKEN"],
    #     revision="b0a2a564e5dc311591886bb375e8d5a1aeaade83",
    # ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2",
    token=os.environ["HF_AUTH_TOKEN"],
)
FAKE_TOK_AROUND_IMAGE = "<fake_token_around_image>"
BOS_TOKEN = PROCESSOR.tokenizer.bos_token
BAD_WORDS_IDS = PROCESSOR.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
EOS_WORDS_IDS = PROCESSOR.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids + [PROCESSOR.tokenizer.eos_token_id]
IMAGE_SEQ_LEN = 64#list(MODELS.values())[0].config.perceiver_config.resampler_n_latents

SYSTEM_PROMPT = [
#     """The following is a conversation between a highly knowledgeable and intelligent visual AI assistant, called Assistant, and a human user, called User. In the following interactions, User and Assistant will converse in natural language, and Assistant will do its best to answer User’s questions. Assistant has the ability to perceive images and reason about the content of visual inputs. Assistant was built to be respectful, polite and inclusive. It knows a lot, and always tells the truth. When prompted with an image, it does not make up facts.

# The conversation begins:""",
#     """\nUser:""",
#     "https://i1.pickpik.com/photos/515/906/355/kittens-cats-pet-cute-preview.jpg",
#     "Describe this image.<end_of_utterance>",
#     """\nAssistant: Five kittens are standing together in the center of the photograph. From the left to right, there is one orange kitten, two kittens white and black stripes, and two brown kittens with an orange head. They are in bright green grass and it looks like they are moving forward.<end_of_utterance>""",
#     "\nUser:How about this image?",
#     "https://cdn.pixabay.com/photo/2017/09/25/13/12/puppy-2785074_1280.jpg",
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
#     "https://i.dailymail.co.uk/i/pix/2011/07/01/article-2010308-0CD22A8300000578-496_634x414.jpg",
#     "<end_of_utterance>",
#     """\nAssistant: There is no dogs in this image. The picture shows a tennis player jumping to volley the ball.<end_of_utterance>""",
]

API_TOKEN = os.getenv("HF_AUTH_TOKEN")
# IDEFICS_LOGO = "https://huggingface.co/spaces/HuggingFaceM4/idefics_playground/resolve/main/IDEFICS_logo.png"
BOT_AVATAR = "IDEFICS_logo.png"


# Model processing utils - these will be handled in the model processor directly ultimately
def convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates a wrong background
    # for transparent images. The call to `alpha_composite` handles this case
    if image.mode == "RGB":
        return image

    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


def custom_transform(x):
    x = convert_to_rgb(x)
    x = to_numpy_array(x)

    height, width = x.shape[:2]
    aspect_ratio = width / height
    if width >= height and width > 980:
        width = 980
        height = int(width / aspect_ratio)
    elif height > width and height > 980:
        height = 980
        width = int(height * aspect_ratio)
    width = max(width, 378)
    height = max(height, 378)

    x = resize(x, (height, width), resample=PILImageResampling.BILINEAR)
    x = PROCESSOR.image_processor.rescale(x, scale=1 / 255)
    x = PROCESSOR.image_processor.normalize(
        x,
        mean=PROCESSOR.image_processor.image_mean,
        std=PROCESSOR.image_processor.image_std
    )
    x = to_channel_dimension_format(x, ChannelDimension.FIRST)
    x = torch.tensor(x)
    return x


def create_model_inputs(
        input_texts: List[str],
        image_lists: List[List[Image.Image]],
    ):
    """
    All this logic will eventually be handled inside the model processor.
    """
    inputs = PROCESSOR.tokenizer(
        input_texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    )

    output_images = [
        [PROCESSOR.image_processor(img, transform=custom_transform) for img in im_list]
        for im_list in image_lists
    ]
    total_batch_size = len(output_images)
    max_num_images = max([len(img_l) for img_l in output_images])
    if max_num_images > 0:
        max_height = max([i.size(2) for img_l in output_images for i in img_l])
        max_width = max([i.size(3) for img_l in output_images for i in img_l])
        padded_image_tensor = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        padded_pixel_attention_masks = torch.zeros(
            total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool
        )
        for batch_idx, img_l in enumerate(output_images):
            for img_idx, img in enumerate(img_l):
                im_height, im_width = img.size()[2:]
                padded_image_tensor[batch_idx, img_idx, :, :im_height, :im_width] = img
                padded_pixel_attention_masks[batch_idx, img_idx, :im_height, :im_width] = True

        inputs["pixel_values"] = padded_image_tensor
        inputs["pixel_attention_mask"] = padded_pixel_attention_masks

    return inputs


# Chatbot utils
def is_image(string: str) -> bool:
    """
    There are two ways for images: local image path or url.
    """
    return is_url(string) or string.startswith(DEFAULT_TEMP_DIR)


def is_url(string: str) -> bool:
    """
    Checks if the passed string contains a valid url and nothing else. e.g. if space is included it's immediately
    invalidated the url
    """
    if " " in string:
        return False
    result = urlparse(string)
    return all([result.scheme, result.netloc])


def prompt_list_to_model_input(prompt_list: List[str]) -> Tuple[str, List[Image.Image]]:
    """
    Create the final input string and image list to feed to the model.
    """
    images = []
    for idx, part in enumerate(prompt_list):
        if is_image(part):
            images.append(Image.open(part))
            prompt_list[idx] = f"{FAKE_TOK_AROUND_IMAGE}{'<image>' * IMAGE_SEQ_LEN}{FAKE_TOK_AROUND_IMAGE}"
    input_text = "".join(prompt_list)
    input_text = input_text.replace(FAKE_TOK_AROUND_IMAGE * 2, FAKE_TOK_AROUND_IMAGE)
    input_text = BOS_TOKEN + input_text.strip()
    return input_text, images


def turn_is_pure_media(turn):
    return turn[1] is None


def format_user_prompt_with_im_history_and_system_conditioning(
    user_prompt, chat_history
) -> List[str]:
    """
    Produces the resulting list that needs to go inside the processor.
    It handles the potential image(s), the history and the system conditionning.
    """
    resulting_list = copy.deepcopy(SYSTEM_PROMPT)

    # Format history
    for turn in chat_history:
        if turn_is_pure_media(turn):
            media = turn[0][0]
            if resulting_list == [] or (resulting_list != [] and resulting_list[-1].endswith("<end_of_utterance>")):
                resulting_list.append("\nUser:")
            resulting_list.append(media)
        else:
            user_utterance, assistant_utterance = turn
            if resulting_list and is_image(resulting_list[-1]): # means that previous `turn` in `chat_history` was a pure media
                resulting_list.append(f"{user_utterance.strip()}<end_of_utterance>\nAssistant: {assistant_utterance}<end_of_utterance>")
            else:
                resulting_list.append(f"\nUser: {user_utterance.strip()}<end_of_utterance>\nAssistant: {assistant_utterance}<end_of_utterance>")

    # Format current input
    if not user_prompt["files"]:
        resulting_list.append(f"\nUser: ")
    else:
        # Choosing to put the image first when the image is inputted through the UI, but this is an arbiratrary choice.
        resulting_list.append("\nUser:")
        resulting_list.extend([im["path"] for im in user_prompt["files"]])
    resulting_list.append(f"{user_prompt['text']}<end_of_utterance>\nAssistant:")

    return resulting_list


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

    formated_prompt_list = format_user_prompt_with_im_history_and_system_conditioning(
        user_prompt=user_prompt,
        chat_history=chat_history,
    )

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
        "bad_words_ids": BAD_WORDS_IDS,
        "eos_token_id": EOS_WORDS_IDS,
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
    input_text, images = prompt_list_to_model_input(formated_prompt_list)
    inputs = create_model_inputs([input_text], [images])
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generation_args.update(inputs)

    # # The regular non streaming generation mode
    # _ = generation_args.pop("streamer")
    # generated_ids = MODELS[model_selector].generate(**generation_args)
    # generated_text = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # return generated_text

    thread = Thread(
        target=MODELS[model_selector].generate,
        kwargs=generation_args,
    )
    thread.start()

    print("start generating")
    acc_text = ""
    try:
        for text_token in streamer:
            acc_text += text_token
            yield acc_text
    except Exception as e:
        print("error")
        gr.Error(e)
    print("success")


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
    value=1.0,
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
    label="IDEFICS2",
    avatar_images=[None, BOT_AVATAR],
)


with gr.Blocks(fill_height=True) as demo:
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

demo.launch()
