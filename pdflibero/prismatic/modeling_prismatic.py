"""
modeling_prismatic.py

Core HuggingFace-style PrismaticPreTrainedModel and PrismaticForConditionalGeneration class definitions, inheriting
from the default `transformers.PretrainedModel`. Meant to be standalone and self-contained, but exactly replicate the
logic in `prismatic.models.vlms.prismatic.py`.

Note =>> for the time being, not adding the custom HF "docstring" formatting.

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

import logging
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import numpy as np
import timm
import tokenizers
import torch
import torch.nn as nn
import transformers
from rich import print as rprint
from timm.models.vision_transformer import LayerScale
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration_prismatic import OpenVLAConfig, PrismaticConfig

# Get Logger
logger = logging.getLogger(__name__)


# === PyTorch/HuggingFace Default IGNORE_INDEX (for CrossEntropyLoss labels)
IGNORE_INDEX = -100


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Prismatic Vision Backbone (nn.Module) Definitions (w/ Fused Backbone Support) ===
class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        image_sizes: List[int],
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer = timm.create_model(
            timm_model_ids[0],
            pretrained=False,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim

        # If `use_fused_vision_backbone` =>> create "beta" featurizer
        if self.use_fused_vision_backbone:
            self.fused_featurizer = timm.create_model(
                timm_model_ids[1],
                pretrained=False,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.featurizer.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        if self.use_fused_vision_backbone:
            for module in self.fused_featurizer.modules():
                if isinstance(module, LayerScale):
                    ls_apply_patch(module)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)
        return torch.cat([patches, patches_fused], dim=2)  # [1, 256, 1024] + [1, 256, 1152]


# === Prismatic Projector (nn.Module) Definitions ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Main HF Class Definitions ===
@dataclass
class PrismaticCausalLMOutputWithPast(ModelOutput):
    """Base class for Prismatic casual (visually-conditioned) language model outputs; also exposes visual features."""

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    # Additions for VLMs
    projector_features: Optional[torch.FloatTensor] = None


class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"
    _supports_flash_attn_2: bool = True

    def _init_weights(self, module: nn.Module) -> None:
        # Important :: this HF ported version is *not* meant for training from scratch; only inference and fine-tuning!
        #   => As such, this init_weights code is not correct; if training VLMs from scratch, use the main codebase at
        #      https://github.com/TRI-ML/prismatic-vlms
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self) -> bool:
        """Check LLM supports SDPA Attention"""
        return self.language_model._supports_sdpa


class PrismaticForConditionalGeneration(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig) -> None:
        super().__init__(config)

        # [Validation] Lightweight Validate on `config` Fields + Dependency Versions
        if config.use_fused_vision_backbone is None:
            raise ValueError("Missing config field `use_fused_vision_backbone`")

        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            logger.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            config.use_fused_vision_backbone, config.image_sizes, config.timm_model_ids, config.timm_override_act_layers
        )

        # Create Multimodal Projector
        self.projector = PrismaticProjector(
            config.use_fused_vision_backbone,
            vision_dim=self.vision_backbone.embed_dim,
            llm_dim=config.text_config.hidden_size,
        )

        # Instantiate LLM Backbone
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.pad_token_id

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings
        self.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_projector_features = output_projector_features if output_projector_features is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert input_ids.shape[0] == 1, "Generation is only currently supported for batch size of 1!"
            assert past_key_values is not None, "You must provide `past_key_values` during cached generation!"
            assert labels is None, "Unexpected key `labels` provided during cached generation!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), "Missing `input_ids` in language-only forward!"
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (inputs_embeds.shape[0] == pixel_values.shape[0]):
            assert past_key_values is None, "Unexpected key `past_key_values` provided during language-only forward!"

            # Visual Feature Extraction
            patch_features = self.vision_backbone(pixel_values)
            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)

            multimodal_embeddings = torch.cat(
                [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]], dim=1
            )
            multimodal_attention_mask = None
            if attention_mask is not None:
                multimodal_attention_mask = torch.cat(
                    [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]], dim=1
                )

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat([labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1)

            # Dispatch to Language Model
            # Ensure embeddings have the same dtype/device as the language model parameters.
            # This prevents dtype-mismatch errors when the LM is loaded in bfloat16
            # while projector/vision produce float32 (common with mixed-precision setups).
            try:
                lm_param = next(self.language_model.parameters())
                lm_dtype = lm_param.dtype
                lm_device = lm_param.device
            except StopIteration:
                lm_dtype = multimodal_embeddings.dtype
                lm_device = multimodal_embeddings.device

            if multimodal_embeddings.dtype != lm_dtype or multimodal_embeddings.device != lm_device:
                multimodal_embeddings = multimodal_embeddings.to(device=lm_device, dtype=lm_dtype)

            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (inputs_embeds.shape[0] != pixel_values.shape[0]):
            raise ValueError("Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!")

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> Dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        if ((input_ids is not None) and (input_ids.shape[0] > 1)) or (
            (inputs_embeds is not None) and (inputs_embeds.shape[0] > 1)
        ):
            raise ValueError("Generation with batch size > 1 is not currently supported!")

        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs

    # Defer to Language Model (all handle this differently, with different return types)
    def _reorder_cache(self, *args, **kwargs) -> Any:
        return self.language_model._reorder_cache(*args, **kwargs)


class OpenVLA(PrismaticForConditionalGeneration):
    config_class: PretrainedConfig = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig) -> None:
        super().__init__(config)
        self.norm_stats = config.norm_stats

        # Compute action bins
        self.bins = np.linspace(-1, 1, config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # Compute vocab size for de-tokenization -- revert added "multiple of"
        self.vocab_size = self.config.text_config.vocab_size - self.config.pad_to_multiple_of

        # perturbation
        self.feature_perturbation_adapter = None
        self.logits_perturbation_adapter = None
        self.pdf = None

    def make_perturbation_logits_adapter(self, action_dim: int = 7, logits_dim: int = 256, scale: float = 0.01) -> None:
        self.logits_perturbation_adapter = nn.Parameter(torch.zeros(1, action_dim, logits_dim) * scale)

    def make_perturbation_feature_adapter(self, action_dim: int = 7, feature_dim: int = 4096, scale: int = 0.01) -> None:
        self.feature_perturbation_adapter = nn.Parameter(torch.zeros(1, action_dim, feature_dim) * scale)

    def make_pdf(self, logits_dim: int = 256, feature_dim: int = 4096) -> None:
        self.pdf = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, logits_dim),
        )

    def _action_logits_to_outputs(
        self,
        action_logits: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        predicted_action_token_ids = action_logits.argmax(-1).squeeze(0).detach().cpu().numpy() + 31744
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return {"actions": actions, "action_token_ids": torch.as_tensor(predicted_action_token_ids - 31744)}

    def get_action_logits_with_perturbation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        perturb_scale: float = 1.0,
        forced_action_token_ids: Optional[torch.LongTensor] = None,
        return_logits: bool = True,
        **kwargs: str,
    ) -> Dict[str, Any]:
        """Generate action-token logits, optionally adding the learnable perturbation head."""
        am = kwargs.get("attention_mask")
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                if isinstance(am, torch.Tensor) and am.shape[0] == input_ids.shape[0]:
                    am = torch.cat((am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)), dim=1)

        action_dim = self.get_action_dim(unnorm_key)
        base_action_logits, perturbations, p_head_inputs = [], [], []
        sequences = input_ids
        past_key_values = None
        for step_idx in range(action_dim):
            step_input_ids = sequences if past_key_values is None else sequences[:, -1:]
            outputs = self.forward(
                input_ids=step_input_ids,
                attention_mask=am,
                pixel_values=kwargs.get("pixel_values"),
                past_key_values=past_key_values,
                use_cache=kwargs.get("use_cache"),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            base_logits = outputs.logits[:, -1, 31744:32000]
            p_head_input = outputs.hidden_states[-1][:, -1, :].float()
            perturbation = torch.zeros_like(base_logits)
            if self.logits_perturbation_adapter is not None:
                perturbation = perturbation + self.logits_perturbation_adapter[:, step_idx, :].to(
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
            if self.pdf is not None:
                perturbation = perturbation + self.pdf(p_head_input).to(
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )

            final_logits = base_logits + perturb_scale * perturbation
            if forced_action_token_ids is None:
                next_token = torch.argmax(final_logits, dim=-1, keepdim=True) + 31744
            else:
                next_token = forced_action_token_ids[:, step_idx : step_idx + 1].to(sequences.device) + 31744
            base_action_logits.append(base_logits)
            perturbations.append(perturbation)
            p_head_inputs.append(p_head_input)
            sequences = torch.cat([sequences, next_token], dim=1)

            if am is not None:
                am = torch.cat([am, torch.ones((input_ids.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1)
            past_key_values = outputs.past_key_values

        base_action_logits = torch.stack(base_action_logits, dim=1)
        perturbations = torch.stack(perturbations, dim=1)
        p_head_inputs = torch.stack(p_head_inputs, dim=1)
        action_logits = base_action_logits + perturb_scale * perturbations
        action_outputs = self._action_logits_to_outputs(action_logits, unnorm_key)
        result = {
            "actions": action_outputs["actions"],
            "action_token_ids": action_outputs["action_token_ids"],
            "perturbations": perturbations,
            "p_head_inputs": p_head_inputs,
        }
        if return_logits:
            result.update({"base_logits": base_action_logits, "logits": action_logits})
        return result

    def get_perturbated_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        **kwargs: str,
    ) -> torch.Tensor:
        return self.get_action_logits_with_perturbation(input_ids=input_ids, unnorm_key=unnorm_key, **kwargs)

    def predict_with_p_head_io(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        perturb_scale: float = 1.0,
        **kwargs: str,
    ) -> Dict[str, Any]:
        """Generate actions and record only P-head inputs/outputs for collection."""
        am = kwargs.get("attention_mask")
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                if isinstance(am, torch.Tensor) and am.shape[0] == input_ids.shape[0]:
                    am = torch.cat((am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)), dim=1)

        action_dim = self.get_action_dim(unnorm_key)
        action_token_ids, p_head_inputs, p_head_outputs = [], [], []
        sequences = input_ids
        past_key_values = None
        for _ in range(action_dim):
            step_input_ids = sequences if past_key_values is None else sequences[:, -1:]
            outputs = self.forward(
                input_ids=step_input_ids,
                attention_mask=am,
                pixel_values=kwargs.get("pixel_values"),
                past_key_values=past_key_values,
                use_cache=kwargs.get("use_cache"),
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            base_logits = outputs.logits[:, -1, 31744:32000]
            p_head_input = outputs.hidden_states[-1][:, -1, :].float()
            p_head_output = torch.zeros_like(base_logits)
            if self.logits_perturbation_adapter is not None:
                p_head_output = p_head_output + self.logits_perturbation_adapter[:, len(action_token_ids), :].to(
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )
            if self.pdf is not None:
                p_head_output = p_head_output + self.pdf(p_head_input).to(
                    device=base_logits.device,
                    dtype=base_logits.dtype,
                )

            next_local_token = torch.argmax(base_logits + perturb_scale * p_head_output, dim=-1, keepdim=True)
            action_token_ids.append(next_local_token.squeeze(1))
            p_head_inputs.append(p_head_input)
            p_head_outputs.append(p_head_output)
            sequences = torch.cat([sequences, next_local_token + 31744], dim=1)

            if am is not None:
                am = torch.cat([am, torch.ones((input_ids.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1)
            past_key_values = outputs.past_key_values

        action_token_ids = torch.stack(action_token_ids, dim=1).squeeze(0)
        p_head_inputs = torch.stack(p_head_inputs, dim=1)
        p_head_outputs = torch.stack(p_head_outputs, dim=1)
        predicted_action_token_ids = action_token_ids.detach().cpu().numpy() + 31744
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return {
            "actions": actions,
            "action_token_ids": action_token_ids.detach().cpu(),
            "p_head_inputs": p_head_inputs,
            "p_head_outputs": p_head_outputs,
        }

    @torch.no_grad()
    def predict(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        **kwargs: str,
    ) -> torch.Tensor:
        am = kwargs.get("attention_mask")
        # Ensure training-time special empty token alignment when using Llama tokenizer convention
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )
            # If an attention_mask was provided in kwargs, keep it consistent with the new token by
            # appending a 1 (attended) token to the attention mask for the same batch.
            if "attention_mask" in kwargs and kwargs["attention_mask"] is not None:
                if isinstance(am, torch.Tensor) and am.shape[0] == input_ids.shape[0]:
                    kwargs["attention_mask"] = torch.cat(
                        (am, torch.ones((am.shape[0], 1), dtype=am.dtype, device=am.device)), dim=1
                    )

        action_dim = self.get_action_dim(unnorm_key)
        action_logit = []
        sequences = input_ids
        past_key_values = None
        for _ in range(action_dim):
            step_input_ids = sequences if past_key_values is None else sequences[:, -1:]
            outputs = self.forward(
                input_ids=step_input_ids,
                attention_mask=am,
                pixel_values=kwargs.get("pixel_values"),
                past_key_values=past_key_values,
                use_cache=kwargs.get("use_cache"),
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            next_logits = outputs.logits[:, -1, :]  # [B, V]
            action_logit.append(next_logits)
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)  # [B, 1]
            sequences = torch.cat([sequences, next_token], dim=1)

            if am is not None:
                am = torch.cat([am, torch.ones((input_ids.shape[0], 1), dtype=am.dtype, device=am.device)], dim=1)
            past_key_values = outputs.past_key_values
        action_logits = torch.stack(action_logit, dim=1)

        # Calculate entropy
        entropy = torch.distributions.Categorical(logits=action_logits).entropy()
        mean_entropy = entropy.mean()
        rprint(f"Overall Entropy: {mean_entropy.item()}")
        predicted_action_token_ids = action_logits.argmax(-1).squeeze(0).cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        action = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return {"actions": action, "logits": action_logits, "entropy": entropy, "mean_entropy": mean_entropy}

    def logits_to_actions(
        self,
        action_logits: torch.Tensor,
        unnorm_key: Optional[str] = None,
    ) -> np.ndarray:
        """Convert action logits to unnormalized continuous actions."""
        predicted_action_token_ids = action_logits.argmax(-1).squeeze(0).cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]
        # Unnormalize actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    def actions_to_logits(
        self,
        actions: np.ndarray,
        unnorm_key: Optional[str] = None,
    ) -> torch.Tensor:
        """Convert unnormalized continuous actions to action logits."""
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        # Normalize actions
        normalized_actions = np.where(
            mask,
            2 * (actions - action_low) / (action_high - action_low) - 1,
            actions,
        )
        # Discretize actions
        discretized_actions = np.digitize(normalized_actions, self.bins) - 1
        discretized_actions = np.clip(discretized_actions, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        predicted_action_token_ids = self.vocab_size - (discretized_actions + 1)
        # Convert to logits
        batch_size, action_dim = predicted_action_token_ids.shape
        action_logits = torch.full(
            (batch_size, action_dim, self.vocab_size),
            fill_value=-1e9,
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        for i in range(batch_size):
            for j in range(action_dim):
                token_id = predicted_action_token_ids[i, j]
                action_logits[i, j, token_id] = 0.0  # Set logit for the predicted token to 0 (highest probability)
        return action_logits

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Get the dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        return self.norm_stats[unnorm_key]["action"]
