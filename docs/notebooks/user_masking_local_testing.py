from post_process.registry import register_step

@register_step(name="user_masking_local")
def masking_step(func, img_path, out_path ,data, **_):
    # custom logic
    print("Running user masking registered step")
    pass
