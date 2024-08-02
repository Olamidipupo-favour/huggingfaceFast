import os
import subprocess
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Union
from PIL import Image
import io
import torch
from threading import Thread
import time
from transformers import Idefics2ForConditionalGeneration
from transformers import AutoProcessor, TextIteratorStreamer

# Install flash attention
subprocess.run(
    "pip install flash-attn --no-build-isolation",
    env={"FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE"},
    shell=True,
)

app = FastAPI()

DEVICE = torch.device("cpu")
MODELS = {
    "idefics2-8b-chatty": Idefics2ForConditionalGeneration.from_pretrained(
        "HuggingFaceM4/idefics2-8b-chatty",
        torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2",
    ).to(DEVICE),
}
PROCESSOR = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")

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

def format_user_prompt_with_im_history_and_system_conditioning(
    user_prompt, chat_history
):
    resulting_messages = copy.deepcopy(SYSTEM_PROMPT)
    resulting_images = []

    for resulting_message in resulting_messages:
        if resulting_message["role"] == "user":
            for content in resulting_message["content"]:
                if content["type"] == "image":
                    resulting_images.append(load_image_from_url(content["image"]))

    for turn in chat_history:
        if not resulting_messages or (resulting_messages and resulting_messages[-1]["role"] != "user"):
            resulting_messages.append({"role": "user", "content": []})

        if isinstance(turn[0], Image.Image):
            resulting_messages[-1]["content"].append({"type": "image"})
            resulting_images.append(turn[0])
        else:
            user_utterance, assistant_utterance = turn
            resulting_messages[-1]["content"].append({"type": "text", "text": user_utterance.strip()})
            resulting_messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_utterance.strip()}]})

    if not user_prompt["files"]:
        resulting_messages.append({"role": "user", "content": [{"type": "text", "text": user_prompt["text"]}]})
    else:
        resulting_messages.append(
            {
                "role": "user",
                "content": [{"type": "image"}] * len(user_prompt["files"]) + [{"type": "text", "text": user_prompt["text"]}],
            }
        )
        resulting_images.extend([Image.open(path) for path in user_prompt["files"]])

    return resulting_messages, resulting_images

def model_inference(user_prompt, chat_history, model_selector, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p):
    if user_prompt["text"].strip() == "" and not user_prompt["files"]:
        return {"error": "Please input a query and optionally image(s)."}

    if user_prompt["text"].strip() == "" and user_prompt["files"]:
        return {"error": "Please input a text query along the image(s)."}

    streamer = TextIteratorStreamer(PROCESSOR.tokenizer, skip_prompt=True, timeout=5.0)

    generation_args = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "streamer": streamer,
    }

    if decoding_strategy == "Greedy":
        generation_args["do_sample"] = False
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    resulting_text, resulting_images = format_user_prompt_with_im_history_and_system_conditioning(user_prompt, chat_history)
    prompt = PROCESSOR.apply_chat_template(resulting_text, add_generation_prompt=True)
    inputs = PROCESSOR(text=prompt, images=resulting_images if resulting_images else None, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    generation_args.update(inputs)

    thread = Thread(target=MODELS[model_selector].generate, kwargs=generation_args)
    thread.start()

    acc_text = ""
    for text_token in streamer:
        time.sleep(0.04)
        acc_text += text_token
        if acc_text.endswith("<end_of_utterance>"):
            acc_text = acc_text[:-18]
        yield acc_text

@app.post("/predict")
async def predict(text: str = Form(...), files: List[UploadFile] = File(None)):
    chat_history = []
    user_prompt = {"text": text, "files": [await file.read() for file in files] if files else []}
    response = model_inference(user_prompt, chat_history, "idefics2-8b-chatty", "Greedy", 0.4, 512, 1.1, 0.8)
    result = ""
    for r in response:
        result += r
    return JSONResponse(content={"response": result})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
