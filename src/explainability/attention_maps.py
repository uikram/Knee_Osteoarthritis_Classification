"""
Attention map utilities for transformer-based models
"""
def get_vit_attention(model, input_tensor):
    """Extracts attention weights from the last ViT block"""
    # Example placeholder, since actual varies by implementation
    # See timm or Hugging Face ViT docs
    attn_weights = model.backbone.blocks[-1].attn.get_attention_map()
    return attn_weights
