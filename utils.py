import os
import torch

from transformers import AutoProcessor
from transformers.image_utils import to_numpy_array, PILImageResampling, ChannelDimension
from transformers.image_transforms import resize, to_channel_dimension_format
from typing import List
from PIL import Image


PROCESSOR = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2",
    token=os.environ["HF_AUTH_TOKEN"],
)


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
