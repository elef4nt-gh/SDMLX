# Bundled MLX Runtime Scope

This directory contains the adapted mflux runtime subset used by SDMLX for
Qwen Image/Edit and FLUX.2 Klein inference. It contains Python source code, not
model weights, converted checkpoints, or generated caches.

Included:

- shared configuration, scheduler, tokenizer, weight, LoRA, latent, and VAE
  helpers;
- Qwen Image/Edit transformer, vision-language encoder, VAE, and variants;
- FLUX.2 Klein transformer, Qwen3 encoder, VAE, latent, and variants;
- FLUX transformer primitives imported directly by FLUX.2.

Not included:

- standalone mflux command-line generators;
- training adapters;
- unsupported model families and legacy FLUX variants;
- product-unreachable diagnostic and callback helpers.

The adapted source remains covered by the bundled mflux license in
`LICENSE.mflux` and the repository's `THIRD_PARTY_NOTICES.md`.
