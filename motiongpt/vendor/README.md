These files come from OpenMotionLab/MotionGPT commit 001aaca8d0ee218fc17f8265d11ac124044fe42f.
They are used only for the VQ decoder. The upstream MIT license is included.
No SMPL models or evaluation datasets are included.

Native MLX T5 equations/cache conventions are adapted from Apple's MIT
ml-explore/mlx-examples/t5/t5.py at commit
796f5b53cab69a3d48a44233ce21aae889e94a08, with padding masks, GELU-new,
cross-attention KV caching, and direct pinned-checkpoint loading added locally.
Its license is retained as LICENSE.MLX-Examples. The source model weights are
not covered by either code license; see ../model-manifest.json.
