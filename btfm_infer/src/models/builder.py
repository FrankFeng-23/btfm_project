from .modules import TransformerEncoder, ProjectionHead
from .ssl_model import MultimodalBTModel

def build_ssl_model(config, device):
    # Note: After enhancement, the number of S2 data channels is fixed at 12 (10 original bands + 2 doy features), and the number of S1 data channels is 4 (2+2)
    ###############Sentinel-2##################
    s2_backbone_ssl = TransformerEncoder(
        band_num=10,
        latent_dim=config['latent_dim'],
        # nhead=16,
        # num_encoder_layers=32,
        # dim_feedforward=512,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=40
    )
    ###############Sentinel-1##################
    s1_backbone_ssl = TransformerEncoder(
        band_num=2,
        latent_dim=config['latent_dim'],
        # nhead=16,
        # num_encoder_layers=32,
        # dim_feedforward=512,
        nhead=8,
        num_encoder_layers=8,
        dim_feedforward=4096,
        dropout=0.1,
        max_seq_len=40
    )
    if config["fusion_method"] == "concat":
        input_dim_for_projector = config['latent_dim']
    else:
        input_dim_for_projector = config['latent_dim']
    projector_ssl = ProjectionHead(input_dim_for_projector,
                                    config["projector_hidden_dim"],
                                    config["projector_out_dim"])
    # Build the SSL model to load the checkpoint
    ssl_model = MultimodalBTModel(s2_backbone_ssl, s1_backbone_ssl, projector_ssl,
                                    fusion_method=config["fusion_method"], return_repr=True).to(device)
    return ssl_model