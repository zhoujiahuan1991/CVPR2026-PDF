"""Utils for evaluating the OpenVLA policy."""

import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from rich import print as rprint
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from pdflibero.prismatic.configuration_prismatic import OpenVLAConfig
from pdflibero.prismatic.modeling_prismatic import OpenVLA
from pdflibero.prismatic.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    # Determine attention implementation based on GPU capability
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            attn_impl = "flash_attention_2"
            print("[*] Loading in BF16 with Flash-Attention 2 Enabled")
        else:
            attn_impl = "sdpa"
            print("[*] Loading in BF16 with SDPA Attention")
    else:
        attn_impl = "sdpa"
        print("[*] Loading in BF16 with SDPA Attention (CPU)")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLA)

    vla = OpenVLA.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # disable gradient
    for _, param in vla.named_parameters():
        param.requires_grad = False
    if getattr(cfg, "p_logits", False):
        vla.make_perturbation_logits_adapter()
        rprint("[TPL] Enabled perturbation logits adapter.")
    if getattr(cfg, "p_feature", False):
        vla.make_perturbation_feature_adapter()
        rprint("[TPF] Enabled perturbation feature adapter.")
    if getattr(cfg, "pdf", False):
        vla.make_pdf()
        rprint("[PDF] Enabled Prismatic Decision Fusion adapter.")

    trainable = sum(p.numel() for p in vla.parameters() if p.requires_grad)
    total = sum(p.numel() for p in vla.parameters())
    rprint(f"Trainable: {trainable:,} / Total: {total:,} ({100 * trainable / total:.2f}%)")

    vla.to(DEVICE)
    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def prepare_inputs(processor, base_vla_name, obs, task_label, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    # rprint("image shape:", image.size)
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    return inputs


def get_image_attention_map(
    vla, processor, base_vla_name, obs, task_label, unnorm_key, image_dir, idx, center_crop=False
):
    """Compute attention-based saliency map for image regions influencing action."""
    import matplotlib.pyplot as plt
    import numpy as np

    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)

        # Crop and then resize back to original size
        image_tf = crop_and_resize(image_tf, crop_scale, batch_size)

        # Convert back to original data type
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image_tf.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Forward pass to get attentions (use_cache=False for full attentions)
    with torch.no_grad():
        outputs = vla.forward(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
            return_dict=True,
        )
    if outputs.attentions is not None:
        rprint("there is attention")
    else:
        rprint("None attention")
    attentions = outputs.attentions  # Tuple of (num_layers,) each (batch, num_heads, seq_len, seq_len)
    # rprint("attention shape", attentions[-1].shape)
    # Assume image patches start after BOS (position 1), and text before
    # Action tokens are the last ACTION_DIM tokens
    action_dim = vla.get_action_dim(unnorm_key)
    # rprint(action_dim)
    seq_len = attentions[0].shape[-1]
    # rprint(seq_len)
    image_start = 1  # After BOS
    image_end = image_start + (seq_len - inputs["input_ids"].shape[1])  # Approximate image length
    rprint(image_start, image_end, inputs["input_ids"].shape[1])
    action_positions = list(range(seq_len - action_dim, seq_len))
    # rprint("action positions", action_positions)
    # Focus on last few layers (cross-attention heavy)
    layer_idx = -1  # Last layer
    attn_layer = attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
    # rprint("attn_layer shape", attn_layer.shape)
    # Average attention from action tokens to image patches
    action_attn = attn_layer[:, action_positions, image_start:image_end].mean(dim=(0, 1))  # (num_image_patches,)
    # rprint("action_attn shape", action_attn.shape)
    # Assume 14x14 patches (196 total, common for ViT)
    patch_grid = int(np.sqrt(action_attn.shape[0]))
    # rprint(patch_grid)
    # rprint(action_attn.shape)
    rprint(11111111111111)
    # # exit()
    # print(type(action_attn))
    # Ensure tensor is cast to a NumPy-supported dtype (float32) before converting.
    # Some models use bfloat16 tensors which cannot be converted directly to NumPy.
    attn_map = action_attn.view(patch_grid, patch_grid).cpu().float().numpy()
    rprint(22222222)
    # Upsample to image size
    from scipy.ndimage import zoom

    img_size = 224  # Assuming resized
    attn_upsampled = zoom(attn_map, img_size / patch_grid, order=1)
    rprint("3333333")
    # Normalize for visualization
    attn_upsampled = (attn_upsampled - attn_upsampled.min()) / (attn_upsampled.max() - attn_upsampled.min())
    rprint("4444444")
    # Overlay on image
    _, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(attn_upsampled, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.savefig(f"{image_dir}/attention_map-{idx}.png", bbox_inches="tight", pad_inches=0)
    # plt.show()
    rprint(55555)
    # exit()
    # return attn_upsampled
