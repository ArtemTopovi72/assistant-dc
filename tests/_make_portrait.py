"""Generate one real portrait to drive P3/P4/P5 edit verification."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import image
p = image.generate_image_with_comfy(
    ctx=None,
    prompt=("studio portrait photograph of a young woman wearing round black sunglasses "
            "and a red knitted beanie hat, plain light-grey background, sharp focus, "
            "85mm lens, professional lighting, photorealistic, high detail"),
    negative_prompt="blurry, lowres, deformed, cartoon, painting",
    steps=8, cfg=1.0, seed=4242, width=768, height=768)
print("PORTRAIT:", p)
