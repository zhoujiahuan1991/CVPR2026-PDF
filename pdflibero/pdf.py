import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import draccus
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from kornia.augmentation import RandomResizedCrop
from rich import print as rprint
from torch.optim import Adam

from pdflibero.utils.libero_utils import (
    benchmark,
    get_additional_norm_stats,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from pdflibero.utils.openvla_utils import get_processor, get_vla, prepare_inputs
from pdflibero.utils.robot_utils import (
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class CollectConfig:
    model_family: str = "openvla"
    algo_name: str = "pdf"
    pretrained_checkpoint: Union[str, Path] = ""
    task_suite_name: str = "libero_spatial"
    task_id: int = -1
    center_crop: bool = True
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    result_dir: str = None
    exp_time: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed: int = 7
    attn_impl: str = None

    # Keep this small. With the default, each state uses original + one augmented view.
    augmentation_times: int = 2
    perturb_scale: float = 1.0

    # P-head must exist so we can collect its inputs/current outputs consistently.
    pdf: bool = True

    # PDF adaptation. Only the P-head is optimized; the VLA backbone and LM head stay frozen.
    learning_rate: float = 1e-4
    kl_coef: float = 0.01
    feedback_baseline: float = 0.5
    baseline_momentum: float = 0.9
    update_batch_size: int = 64
    update_epochs: int = 1

    save_videos: bool = True


def _max_steps(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unsupported task suite: {task_suite_name}")


def _majority_vote(local_action_token_ids: List[torch.Tensor]) -> np.ndarray:
    token_array = torch.stack(local_action_token_ids, dim=0).detach().cpu().numpy()
    voted = []
    for dim_tokens in token_array.T:
        values, counts = np.unique(dim_tokens, return_counts=True)
        max_count = counts.max()
        winners = values[counts == max_count]
        if winners.size == 1:
            voted.append(int(winners[0]))
            continue
        # With one augmented view, ties are common. Prefer the augmented view when it disagrees
        # so successful augmented rollouts can provide useful P-head examples.
        voted.append(int(dim_tokens[-1]))
    return np.asarray(voted, dtype=np.int64)


def _local_token_ids_to_actions(model, local_token_ids: np.ndarray, unnorm_key: str) -> np.ndarray:
    predicted_action_token_ids = local_token_ids + 31744
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    normalized_actions = model.bin_centers[discretized_actions]
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def _make_observation(obs, image):
    return {
        "full_image": image,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }


def _run_episode(
    *,
    cfg: CollectConfig,
    model,
    processor,
    env,
    initial_state,
    task_description: str,
    resize_size: int,
    transform,
    use_augmentation: bool,
) -> Tuple[bool, List[np.ndarray], List[Dict[str, Any]]]:
    env.reset()
    obs = env.set_init_state(initial_state)
    done = False
    t = 0
    replay_images = []
    samples = []

    while t < _max_steps(cfg.task_suite_name) + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        img = get_libero_image(obs, resize_size)
        replay_images.append(img)
        observation = _make_observation(obs, img)
        inputs = prepare_inputs(processor, cfg.pretrained_checkpoint, observation, task_description, cfg.center_crop)

        with torch.no_grad():
            outputs = model.predict_with_p_head_io(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                perturb_scale=cfg.perturb_scale,
                use_cache=True,
            )

        candidate_tokens = [outputs["action_token_ids"]]
        candidate_outputs = [outputs]

        if use_augmentation:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for _ in range(max(0, cfg.augmentation_times - 1)):
                aug_img_tensor = transform(img_tensor).squeeze(0).clamp(0.0, 1.0)
                aug_img = (aug_img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                aug_observation = _make_observation(obs, aug_img)
                aug_inputs = prepare_inputs(
                    processor,
                    cfg.pretrained_checkpoint,
                    aug_observation,
                    task_description,
                    cfg.center_crop,
                )
                with torch.no_grad():
                    aug_outputs = model.predict_with_p_head_io(
                        **aug_inputs,
                        unnorm_key=cfg.unnorm_key,
                        perturb_scale=cfg.perturb_scale,
                        use_cache=True,
                    )
                candidate_tokens.append(aug_outputs["action_token_ids"])
                candidate_outputs.append(aug_outputs)

        voted_tokens_np = _majority_vote(candidate_tokens)
        voted_tokens = torch.as_tensor(voted_tokens_np, dtype=outputs["action_token_ids"].dtype)
        selected_index = len(candidate_outputs) - 1
        selected_outputs = candidate_outputs[selected_index]

        samples.append(
            {
                "p_head_inputs": selected_outputs["p_head_inputs"].squeeze(0).detach().cpu(),
                "p_head_outputs": selected_outputs["p_head_outputs"].squeeze(0).detach().cpu(),
                "original_action_token_ids": outputs["action_token_ids"].detach().cpu(),
                "target_action_token_ids": voted_tokens.detach().cpu(),
                "changed_by_augmentation": not torch.equal(outputs["action_token_ids"].detach().cpu(), voted_tokens),
            }
        )

        actions = _local_token_ids_to_actions(model, voted_tokens_np, cfg.unnorm_key)
        actions = normalize_gripper_action(actions.copy(), binarize=True)
        if cfg.model_family == "openvla":
            actions = invert_gripper_action(actions)

        obs, _, done, _ = env.step(actions.tolist())
        if done:
            break
        t += 1

    return bool(done), replay_images, samples


def _save_success_samples(cfg, task_id, episode_idx, task_description, samples):
    if not samples:
        return None

    save_path = os.path.join(cfg.result_dir, f"task-{task_id}-episode-{episode_idx}-aug-success.pt")
    torch.save(
        {
            "task_id": task_id,
            "episode_idx": episode_idx,
            "task_description": task_description,
            "p_head_inputs": torch.stack([sample["p_head_inputs"] for sample in samples], dim=0),
            "p_head_outputs": torch.stack([sample["p_head_outputs"] for sample in samples], dim=0),
            "original_action_token_ids": torch.stack([sample["original_action_token_ids"] for sample in samples], dim=0),
            "target_action_token_ids": torch.stack([sample["target_action_token_ids"] for sample in samples], dim=0),
            "changed_by_augmentation": torch.as_tensor(
                [sample["changed_by_augmentation"] for sample in samples],
                dtype=torch.bool,
            ),
        },
        save_path,
    )
    return save_path


def _get_lm_head(model):
    if hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head
    if hasattr(model.language_model, "model") and hasattr(model.language_model.model, "lm_head"):
        return model.language_model.model.lm_head
    raise AttributeError("Could not find language model LM head for PDF loss.")


def _optimize_p_head(model, optimizer, samples, feedback, baseline, cfg):
    if optimizer is None or model.pdf is None or not samples:
        return None

    advantage = float(feedback - baseline)
    if advantage <= 0.0:
        return None

    p_head_inputs = torch.stack([sample["p_head_inputs"] for sample in samples], dim=0).to(next(model.parameters()).device)
    target_tokens = torch.stack([sample["target_action_token_ids"] for sample in samples], dim=0).to(p_head_inputs.device)
    flat_inputs = p_head_inputs.reshape(-1, p_head_inputs.shape[-1])
    flat_targets = target_tokens.reshape(-1)

    lm_head = _get_lm_head(model)
    batch_size = min(cfg.update_batch_size, flat_inputs.shape[0])
    last_loss = None

    model.pdf.train()
    for _ in range(cfg.update_epochs):
        sample_ids = torch.randperm(flat_inputs.shape[0], device=flat_inputs.device)[:batch_size]
        batch_inputs = flat_inputs[sample_ids]
        batch_targets = flat_targets[sample_ids]

        with torch.no_grad():
            base_logits = lm_head(batch_inputs.to(dtype=next(lm_head.parameters()).dtype))[:, 31744:32000].float()
            base_probs = F.softmax(base_logits, dim=-1)

        perturbation = model.pdf(batch_inputs.float()).float()
        perturbed_logits = base_logits + cfg.perturb_scale * perturbation
        perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)
        selected_log_probs = perturbed_log_probs.gather(-1, batch_targets.view(-1, 1)).squeeze(-1)

        reinforce_loss = -advantage * selected_log_probs.mean()
        kl_loss = F.kl_div(perturbed_log_probs, base_probs, reduction="batchmean")
        loss = reinforce_loss + cfg.kl_coef * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    model.pdf.eval()
    return last_loss


@draccus.wrap()
def collect(cfg: CollectConfig) -> None:
    cfg.result_dir = (
        f"./results/{cfg.task_suite_name}/{cfg.algo_name}-{cfg.pretrained_checkpoint.split('/')[-1]}-{cfg.exp_time}"
    )
    os.makedirs(cfg.result_dir, exist_ok=True)
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_vla(cfg)
    model.norm_stats.update(get_additional_norm_stats())
    if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
        cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
    assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA norm_stats"
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("pdf.")
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = Adam(trainable_params, lr=cfg.learning_rate) if trainable_params else None
    processor = get_processor(cfg)

    log_path = os.path.join(cfg.result_dir, "logs.txt")
    failure_path = os.path.join(cfg.result_dir, "failures.jsonl")
    log_file = open(log_path, "w")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task_ids = [cfg.task_id] if cfg.task_id >= 0 else list(range(task_suite.n_tasks))
    resize_size = get_image_resize_size(cfg)
    transform = RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33))

    total_episodes, total_successes, aug_successes, plain_successes, failures = 0, 0, 0, 0, 0
    baseline = cfg.feedback_baseline
    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            initial_state = initial_states[episode_idx]
            rprint(f"\nTask {task_id}, episode {episode_idx}: {task_description}")
            log_file.write(f"\nTask {task_id}, episode {episode_idx}: {task_description}\n")

            aug_done, aug_replay_images, aug_samples = _run_episode(
                cfg=cfg,
                model=model,
                processor=processor,
                env=env,
                initial_state=initial_state,
                task_description=task_description,
                resize_size=resize_size,
                transform=transform,
                use_augmentation=True,
            )

            saved_path = None
            pdf_loss = None
            plain_done = None
            if aug_done:
                aug_successes += 1
                saved_path = _save_success_samples(cfg, task_id, episode_idx, task_description, aug_samples)
                pdf_loss = _optimize_p_head(model, optimizer, aug_samples, feedback=1.0, baseline=baseline, cfg=cfg)
                if cfg.save_videos:
                    save_rollout_video(
                        aug_replay_images,
                        task_id,
                        episode_idx,
                        success=True,
                        task_description=f"aug_success_{task_description}",
                        video_dir=cfg.result_dir,
                        log_file=log_file,
                    )
            else:
                plain_done, plain_replay_images, _ = _run_episode(
                    cfg=cfg,
                    model=model,
                    processor=processor,
                    env=env,
                    initial_state=initial_state,
                    task_description=task_description,
                    resize_size=resize_size,
                    transform=transform,
                    use_augmentation=False,
                )
                if plain_done:
                    plain_successes += 1
                    if cfg.save_videos:
                        save_rollout_video(
                            plain_replay_images,
                            task_id,
                            episode_idx,
                            success=True,
                            task_description=f"plain_success_{task_description}",
                            video_dir=cfg.result_dir,
                            log_file=log_file,
                        )
                else:
                    failures += 1
                    with open(failure_path, "a") as failure_file:
                        failure_file.write(
                            json.dumps(
                                {
                                    "task_id": task_id,
                                    "episode_idx": episode_idx,
                                    "task_description": task_description,
                                    "augmentation_success": False,
                                    "plain_success": False,
                                }
                            )
                            + "\n"
                        )

            final_success = aug_done or bool(plain_done)
            baseline = cfg.baseline_momentum * baseline + (1.0 - cfg.baseline_momentum) * float(final_success)
            total_episodes += 1
            total_successes += int(final_success)
            success_rate = total_successes / total_episodes * 100.0
            plain_status = "N/A" if plain_done is None else plain_done

            message = (
                f"Success: {final_success} | aug_success: {aug_done} | plain_success: {plain_status} | "
                f"saved: {saved_path} | pdf_loss: {pdf_loss} | "
                f"total: {total_successes}/{total_episodes} ({success_rate:.1f}%)"
            )
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

    summary = (
        f"Final success: {total_successes}/{total_episodes} "
        f"({(total_successes / total_episodes * 100.0) if total_episodes else 0.0:.1f}%), "
        f"aug_successes={aug_successes}, plain_successes={plain_successes}, failures={failures}"
    )
    print(summary)
    log_file.write(summary + "\n")
    log_file.close()


if __name__ == "__main__":
    collect()
