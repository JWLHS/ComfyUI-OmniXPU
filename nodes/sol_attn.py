"""Opt-in DG2 Sol-Attn patch node for MiniMax H3 self-attention.

Sol-Attn is sparse approximate attention (quality tradeoff), so it is
explicitly opt-in: place this node after the model loader to route MiniMax
H3 self-attention through ``omni_xpu_kernel.cute.sol_attn`` (two-phase:
DPAS summary + FMA exact). Without the node the original attention path is
unchanged. Short sequences (below ``min_sequence``) and non-bf16 fall back
to the original forward.
"""


class OmniXPUPatchSolAttn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "tau": ("FLOAT", {"default": 1.3, "min": 0.1, "max": 4.0,
                                  "step": 0.1}),
                "min_sequence": ("INT", {"default": 16384, "min": 1024,
                                         "max": 131072, "step": 1024}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "OmniXPU"

    def patch(self, model, tau=1.3, min_sequence=16384):
        from comfy import model_management as mm
        from comfy.quant_ops import ck as _ck
        from omni_xpu_kernel import cute

        if not cute.supports_sol_attn():
            raise RuntimeError(
                "Sol-Attn backend unavailable: install the omni_xpu_kernel "
                "build that packages the lgrf_sol_attn_sum/exact sidecars"
            )

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if blocks is None:
            raise RuntimeError(
                "No MiniMax H3 transformer blocks found; "
                "OmniXPU Patch Sol-Attn only supports MiniMax H3"
            )

        patched = 0
        for idx, block in enumerate(blocks):
            attn = getattr(block, "self_attn", None)
            if attn is None:
                continue
            bound = _make_sol_attn_forward(
                attn, cute, _ck, mm, float(tau), int(min_sequence)
            ).__get__(attn, attn.__class__)
            model_clone.add_object_patch(
                f"diffusion_model.blocks.{idx}.self_attn.forward",
                bound)
            patched += 1
        if patched == 0:
            raise RuntimeError(
                "No MiniMax H3 self-attention modules found to patch"
            )
        return (model_clone,)


def _make_sol_attn_forward(attn, cute, ck, mm, tau, min_sequence):
    """Return the patched self-attention forward (bound via __get__)."""
    orig_forward = attn.forward

    def forward(self, x, rope_freqs=None, transformer_options=None):
        import torch
        x_orig = x
        if isinstance(x, list):
            x = x.pop()
        dtype = x.dtype
        device = x.device
        s = x.shape[0]
        # 短序列或非 bf16：回退原 forward（质量/契约保护）
        if s < min_sequence or dtype != torch.bfloat16:
            return orig_forward(x_orig, rope_freqs=rope_freqs,
                                transformer_options=transformer_options or {})
        q, k, v = self.qkv_proj(x).split(
            self.heads * self.head_dim, dim=-1)
        del x
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        v = v.view(1, s, self.heads, self.head_dim)
        if rope_freqs is not None:
            qw = mm.cast_to(self.q_norm.weight, device=device)
            kw = mm.cast_to(self.k_norm.weight, device=device)
            ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw,
                epsilon=self.q_norm.eps,
                rot_dim=rope_freqs.shape[-3] * 2)
        else:
            q = self.q_norm(q)
            k = self.k_norm(k)
        o = cute.sol_attn(q, k, v, tau=tau)
        return self.out_proj(o.view(s, self.heads * self.head_dim))

    return forward


NODE_CLASS_MAPPINGS = {
    "OmniXPUPatchSolAttn": OmniXPUPatchSolAttn,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OmniXPUPatchSolAttn": "OmniXPU Patch Sol-Attn",
}
