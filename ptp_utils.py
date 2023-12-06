# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict

# from IPython.display import display
from tqdm.notebook import tqdm
from diffusers.utils.torch_utils import randn_tensor


def text_under_image(
    image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)
):
    h, w, c = image.shape
    offset = int(h * 0.2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y), font, 1, text_color, 2)
    return img


def view_images(images, filename, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = (
        np.ones(
            (
                h * num_rows + offset * (num_rows - 1),
                w * num_cols + offset * (num_cols - 1),
                3,
            ),
            dtype=np.uint8,
        )
        * 255
    )
    for i in range(num_rows):
        for j in range(num_cols):
            image_[
                i * (h + offset) : i * (h + offset) + h :,
                j * (w + offset) : j * (w + offset) + w,
            ] = images[i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(filename)
    # display(pil_img)


def diffusion_step(
    model, controller, latents, context, t, guidance_scale, low_resource=False
):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])[
            "sample"
        ]
        noise_prediction_text = model.unet(
            latents, t, encoder_hidden_states=context[1]
        )["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        # latents_input = latents_input.to(torch.float16)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)["sample"]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def image2imagecpu(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(
        batch_size, model.unet.in_channels, height // 8, width // 8
    ).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control_SD(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)

    # set timesteps
    # extra_set_kwargs = {"offset": 1}
    # model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    model.scheduler.set_timesteps(num_inference_steps)
    # for t in list(model.scheduler.timesteps.detach().cpu().numpy()):
    for t in reversed(range(0, 1000, 20)):
        latents = diffusion_step(
            model, controller, latents, context, t, guidance_scale, low_resource
        )
        view_images(
            latent2image(model.vae, latents),
            filename="%s.jpg" % (str(t).zfill(4)),
        )

    image = latent2image(model.vae, latents)
    return image, latent


@torch.no_grad()
def text2image_deepfloyd_stage1(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control_DF(model, controller, True)
    height = width = 64

    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]

    if not low_resource:
        context = torch.cat(context)

    latent = torch.randn(
        (1, model.unet.in_channels, height, width),
        generator=generator,
    )
    latents = latent.expand(batch_size, model.unet.in_channels, height, width).to(
        model.device
    )

    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    for t in reversed(
        range(0, 1000, 20)
    ):  # list(model.scheduler.timesteps.detach().cpu().numpy())
        # latents: B 3 H W -> B*2 3 H W
        # Version 1
        unet_call = model.unet(
            torch.cat([latents] * 2).to(torch.float16), t, encoder_hidden_states=context
        )
        noise_pred = unet_call["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        scheduler_step = model.scheduler.step(noise_pred, t, latents)
        latents = scheduler_step["prev_sample"]
        latents = controller.step_callback(latents)

        # view_images(
        #     image2imagecpu(latents),
        #     filename="%s_%s_stage1.jpg" % (prompt[0][:20], str(t).zfill(4)),
        # )

    return latents, image2imagecpu(latents), context


@torch.no_grad()
def text2image_deepfloyd_stage2(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    context: Optional[torch.FloatTensor] = None,
):
    register_attention_control_DF(model, controller, False)
    height = width = 256
    noise_level = 250

    batch_size = len(prompt)

    latents = latents.to(model.device)
    context = context.to(model.device)

    # set timesteps
    model.scheduler.set_timesteps(num_inference_steps)
    upscaled = F.interpolate(
        latents, (height, width), mode="bilinear", align_corners=True
    )
    noise_level = torch.tensor(
        [noise_level] * upscaled.shape[0], device=upscaled.device
    )
    noise = randn_tensor(
        upscaled.shape,
        generator=generator,
        device=upscaled.device,
        dtype=upscaled.dtype,
    )
    upscaled = model.image_noising_scheduler.add_noise(
        upscaled, noise, timesteps=noise_level
    )
    noise_level = torch.cat([noise_level] * 2)
    intermediate_images = model.prepare_intermediate_images(
        batch_size * 1,
        model.unet.config.in_channels // 2,
        height,
        width,
        context.dtype,
        model.device,
        generator,
    )

    for t in reversed(
        range(0, 1000, 20)
    ):  # list(model.scheduler.timesteps.detach().cpu().numpy())
        latents = torch.cat([intermediate_images, upscaled], dim=1)
        latents = torch.cat([latents] * 2)
        latents = model.scheduler.scale_model_input(latents, t)
        unet_call = model.unet(
            latents.to(torch.float16),
            t,
            encoder_hidden_states=context,
            class_labels=noise_level,
        )
        # noise_pred = unet_call["sample"]
        # noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + guidance_scale * (
        #     noise_prediction_text - noise_pred_uncond
        # )

        noise_pred = unet_call[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(latents.shape[1] // 2, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(
            latents.shape[1] // 2, dim=1
        )
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        intermediate_images = model.scheduler.step(noise_pred, t, intermediate_images)[
            0
        ]
        # scheduler_step = model.scheduler.step(noise_pred, t, latents)
        # latents = scheduler_step["prev_sample"]
        intermediate_images = controller.step_callback(intermediate_images)

        # view_images(
        #     image2imagecpu(intermediate_images),
        #     filename="%s_%s_stage2.jpg" % (prompt[0][:20], str(t).zfill(4)),
        # )

    return image2imagecpu(intermediate_images)


@torch.no_grad()
def text2image_deepfloyd_debug(
    model_stage1,
    model_stage2,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    register_attention_control_DF(model_stage1, controller)
    # register_attention_control_DF(model_stage2, controller)

    height_1 = width_1 = 64
    height_2 = width_2 = 256

    batch_size = len(prompt)

    text_input = model_stage1.tokenizer(
        prompt,
        padding="max_length",
        max_length=model_stage1.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model_stage1.text_encoder(
        text_input.input_ids.to(model_stage1.device)
    )[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = model_stage1.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = model_stage1.text_encoder(
        uncond_input.input_ids.to(model_stage1.device)
    )[0]

    context = [uncond_embeddings, text_embeddings]

    if not low_resource:
        context = torch.cat(context)

    latent = torch.randn(
        (1, model_stage1.unet.in_channels, height_1, width_1),
        generator=generator,
    )
    latents = latent.expand(
        batch_size, model_stage1.unet.in_channels, height_1, width_1
    ).to(model_stage1.device)

    # set timesteps
    model_stage1.scheduler.set_timesteps(num_inference_steps)
    for t in reversed(
        range(0, 1000, 20)
    ):  # list(model.scheduler.timesteps.detach().cpu().numpy())
        unet_call = model_stage1.unet(
            torch.cat([latents] * 2).to(torch.float16), t, encoder_hidden_states=context
        )
        noise_pred = unet_call["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        scheduler_step = model_stage1.scheduler.step(noise_pred, t, latents)
        latents = scheduler_step["prev_sample"]
        latents = controller.step_callback(latents)

        view_images(
            image2imagecpu(latents),
            filename="%s_%s_stage1.jpg" % (prompt[0][:20], str(t).zfill(4)),
        )

    image_stage1 = image2imagecpu(latents)

    # set timesteps
    model_stage2.scheduler.set_timesteps(num_inference_steps)
    upscaled = F.interpolate(
        latents, (height_2, width_2), mode="bilinear", align_corners=True
    )
    noise_level = 250
    noise_level = torch.tensor(
        [noise_level] * upscaled.shape[0], device=upscaled.device
    )
    noise = randn_tensor(
        upscaled.shape,
        generator=generator,
        device=upscaled.device,
        dtype=upscaled.dtype,
    )
    upscaled = model_stage2.image_noising_scheduler.add_noise(
        upscaled, noise, timesteps=noise_level
    )
    noise_level = torch.cat([noise_level] * 2)
    intermediate_images = model_stage2.prepare_intermediate_images(
        batch_size * 1,
        model_stage2.unet.config.in_channels // 2,
        height_2,
        width_2,
        context.dtype,
        latents.device,
        generator,
    )

    for t in reversed(
        range(0, 1000, 20)
    ):  # list(model.scheduler.timesteps.detach().cpu().numpy())
        latents = torch.cat([intermediate_images, upscaled], dim=1)
        latents = torch.cat([latents] * 2)
        latents = model_stage2.scheduler.scale_model_input(latents, t)
        unet_call = model_stage2.unet(
            latents.to(torch.float16),
            t,
            encoder_hidden_states=context,
            class_labels=noise_level,
        )
        # noise_pred = unet_call["sample"]
        # noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        # noise_pred = noise_pred_uncond + guidance_scale * (
        #     noise_prediction_text - noise_pred_uncond
        # )

        noise_pred = unet_call[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(latents.shape[1] // 2, dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(
            latents.shape[1] // 2, dim=1
        )
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        noise_pred = torch.cat([noise_pred, predicted_variance], dim=1)

        intermediate_images = model_stage2.scheduler.step(
            noise_pred, t, intermediate_images
        )[0]
        # scheduler_step = model_stage2.scheduler.step(noise_pred, t, latents)
        # latents = scheduler_step["prev_sample"]
        intermediate_images = controller.step_callback(intermediate_images)

        view_images(
            image2imagecpu(intermediate_images),
            filename="%s_%s_stage2.jpg" % (prompt[0][:20], str(t).zfill(4)),
        )

    image_stage2 = image2imagecpu(intermediate_images)
    return image_stage1, image_stage2


def register_attention_control_SD(model, controller):
    def ca_forward_new(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
        ):
            is_cross = encoder_hidden_states is not None
            print("=====> ####### is_cross = ", is_cross)

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(
                    batch_size, channel, height * width
                ).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape
                if encoder_hidden_states is None
                else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )

            if self.group_norm is not None:
                hidden_states = self.group_norm(
                    hidden_states.transpose(1, 2)
                ).transpose(1, 2)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ in ["Attention"]:
            """
            CrossAttnDownBlock2D
            CrossAttnUpBlock2D
            Attention
            "CrossAttention":
            """
            net_.forward = ca_forward_new(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def register_attention_control_DF(model, controller, is_cross):
    def ca_forward_v2(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
        ):
            # self.only_cross_attention = False
            # is_cross = encoder_hidden_states is not None
            scale = 1.0

            residual = hidden_states

            hidden_states = hidden_states.view(
                hidden_states.shape[0], hidden_states.shape[1], -1
            ).transpose(1, 2)

            batch_size, sequence_length, _ = hidden_states.shape

            attention_mask = self.prepare_attention_mask(
                attention_mask, sequence_length, batch_size, out_dim=4
            )

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(
                    encoder_hidden_states
                )

            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

            query = self.to_q(hidden_states)
            query = self.head_to_batch_dim(query, out_dim=4)

            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.head_to_batch_dim(
                encoder_hidden_states_key_proj, out_dim=4
            )
            encoder_hidden_states_value_proj = self.head_to_batch_dim(
                encoder_hidden_states_value_proj, out_dim=4
            )

            if not self.only_cross_attention:
                key = self.to_k(hidden_states)
                value = self.to_v(hidden_states)
                key = self.head_to_batch_dim(key, out_dim=4)
                value = self.head_to_batch_dim(value, out_dim=4)
                key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
                value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
            else:
                key = encoder_hidden_states_key_proj
                value = encoder_hidden_states_value_proj

            attention_probs = self.get_attention_scores(
                query.reshape(-1, query.shape[2], query.shape[3]),
                key.reshape(-1, key.shape[2], key.shape[3]),
                attention_mask,
            )
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            # hidden_states = torch.bmm(attention_probs, value)
            # hidden_states = self.batch_to_head_dim(hidden_states)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, residual.shape[1]
            )

            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
            hidden_states = hidden_states + residual

            return hidden_states

        return forward

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == "Attention":
            net_.forward = ca_forward_v2(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, "children"):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count


def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [
            tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)
        ][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(
    alpha,
    bounds: Union[float, Tuple[float, float]],
    prompt_ind: int,
    word_inds: Optional[torch.Tensor] = None,
):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[:start, prompt_ind, word_inds] = 0
    alpha[start:end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(
    prompts,
    num_steps,
    cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
    tokenizer,
    max_num_words=77,
):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0.0, 1.0)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(
            alpha_time_words, cross_replace_steps["default_"], i
        )
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [
                get_word_inds(prompts[i], key, tokenizer)
                for i in range(1, len(prompts))
            ]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(
                        alpha_time_words, item, i, ind
                    )
    alpha_time_words = alpha_time_words.reshape(
        num_steps + 1, len(prompts) - 1, 1, 1, max_num_words
    )
    return alpha_time_words
