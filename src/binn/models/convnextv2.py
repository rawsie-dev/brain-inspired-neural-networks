import timm


def patch_mlp_activations(model, activation_factory):
    patched = 0
    for stage in model.stages:
        for block in stage.blocks:
            block.mlp.act = activation_factory(block.mlp.fc1.out_features)
            patched += 1
    return patched

def build_convnextv2(num_outputs, activation_factory=None, variant="convnextv2_base"):
    model = timm.create_model(variant, pretrained=False, num_classes=num_outputs)
    if activation_factory:
        patch_mlp_activations(model, activation_factory)
    return model
