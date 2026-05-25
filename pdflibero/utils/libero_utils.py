"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os

import imageio
import numpy as np
import tensorflow as tf

libero_source_dir = os.environ.get("LIBERO_SOURCE_DIR")
if libero_source_dir:
    import sys

    sys.path.append(libero_source_dir)
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(
    rollout_images, task_id, episode_idx, success, task_description, video_dir, log_file=None, prefix=""
):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{video_dir}/{prefix}task-{task_id}-episode-{episode_idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_additional_norm_stats():
    return {
        "libero_spatial": {
            "action": {
                "mask": [True, True, True, True, True, True, False],
                "max": [0.9375, 0.9375, 0.9375, 0.1971428543329239, 0.33642858266830444, 0.375, 1.0],
                "mean": [
                    0.15312479436397552,
                    0.13707277178764343,
                    -0.15526802837848663,
                    -0.005176450591534376,
                    -0.01120874285697937,
                    -0.020194264128804207,
                    0.4578818082809448,
                ],
                "min": [-0.9375, -0.9375, -0.9375, -0.1875, -0.3675000071525574, -0.36000001430511475, 0.0],
                "q01": [
                    -0.7454732114076613,
                    -0.6616071462631226,
                    -0.9375,
                    -0.1071428582072258,
                    -0.20678570866584778,
                    -0.1842857152223587,
                    0.0,
                ],
                "q99": [
                    0.9375,
                    0.8758928775787354,
                    0.9321428537368774,
                    0.1039285734295845,
                    0.17678570747375488,
                    0.14571428298950195,
                    1.0,
                ],
                "std": [
                    0.41272708773612976,
                    0.34724321961402893,
                    0.50869220495224,
                    0.037266165018081665,
                    0.07244449853897095,
                    0.05762382969260216,
                    0.49827873706817627,
                ],
            },
            "num_trajectories": 432,
            "num_transitions": 52970,
            "proprio": {
                "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "libero_object": {
            "action": {
                "mask": [True, True, True, True, True, True, False],
                "max": [
                    0.9375,
                    0.8919642567634583,
                    0.9375,
                    0.17678570747375488,
                    0.35035714507102966,
                    0.1810714304447174,
                    1.0,
                ],
                "mean": [
                    0.07096529006958008,
                    0.13498851656913757,
                    -0.04601382836699486,
                    0.00123520044144243,
                    0.006998839322477579,
                    -0.015027612447738647,
                    0.46428999304771423,
                ],
                "min": [
                    -0.8839285969734192,
                    -0.9375,
                    -0.9375,
                    -0.15000000596046448,
                    -0.29035714268684387,
                    -0.32892856001853943,
                    0.0,
                ],
                "q01": [
                    -0.5383928418159485,
                    -0.8758928775787354,
                    -0.9375,
                    -0.06964285671710968,
                    -0.11678571254014969,
                    -0.15964286029338837,
                    0.0,
                ],
                "q99": [
                    0.8464285731315613,
                    0.84375,
                    0.9375,
                    0.08142857253551483,
                    0.14892856776714325,
                    0.0867857113480568,
                    1.0,
                ],
                "std": [
                    0.2681235373020172,
                    0.43846824765205383,
                    0.4474974274635315,
                    0.024446550756692886,
                    0.049355510622262955,
                    0.042107198387384415,
                    0.49879148602485657,
                ],
            },
            "num_trajectories": 454,
            "num_transitions": 66984,
            "proprio": {
                "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "libero_goal": {
            "action": {
                "mask": [True, True, True, True, True, True, False],
                "max": [0.9375, 0.9375, 0.9375, 0.3557142913341522, 0.375, 0.375, 1.0],
                "mean": [
                    0.04721052572131157,
                    0.028835246339440346,
                    -0.1485840231180191,
                    -0.0025010062381625175,
                    0.026408178731799126,
                    0.027379808947443962,
                    0.6299911737442017,
                ],
                "min": [-0.9375, -0.9375, -0.9375, -0.2582142949104309, -0.375, -0.2871428430080414, 0.0],
                "q01": [
                    -0.8785714507102966,
                    -0.7553571462631226,
                    -0.9375,
                    -0.1510714292526245,
                    -0.1639285683631897,
                    -0.13777500048279764,
                    0.0,
                ],
                "q99": [0.9375, 0.9107142686843872, 0.9375, 0.20357142388820648, 0.26357144117355347, 0.375, 1.0],
                "std": [
                    0.3968801498413086,
                    0.3473387360572815,
                    0.49239858984947205,
                    0.055331431329250336,
                    0.07844757288694382,
                    0.10008802264928818,
                    0.48270025849342346,
                ],
            },
            "num_trajectories": 428,
            "num_transitions": 52042,
            "proprio": {
                "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
        "libero_10": {
            "action": {
                "mask": [True, True, True, True, True, True, False],
                "max": [0.9375, 0.9375, 0.9375, 0.30000001192092896, 0.29357144236564636, 0.375, 1.0],
                "mean": [
                    0.01820324920117855,
                    0.05858374014496803,
                    -0.05592384561896324,
                    0.004626928828656673,
                    0.00289608770981431,
                    -0.007673131301999092,
                    0.5457824468612671,
                ],
                "min": [-0.9375, -0.9375, -0.9375, -0.23642857372760773, -0.3053571283817291, -0.3675000071525574, 0.0],
                "q01": [
                    -0.6348214149475098,
                    -0.7741071581840515,
                    -0.7633928656578064,
                    -0.09749999642372131,
                    -0.14819999992847435,
                    -0.2742857038974762,
                    0.0,
                ],
                "q99": [
                    0.7714285850524902,
                    0.8464285731315613,
                    0.9375,
                    0.13928571343421936,
                    0.15964286029338837,
                    0.3246428668498993,
                    1.0,
                ],
                "std": [
                    0.2825464606285095,
                    0.35904666781425476,
                    0.3673802614212036,
                    0.03770702704787254,
                    0.05429719388484955,
                    0.08725254982709885,
                    0.49815231561660767,
                ],
            },
            "num_trajectories": 379,
            "num_transitions": 101469,
            "proprio": {
                "max": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "min": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q01": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "q99": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        },
    }
