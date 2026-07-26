"""TESSERA v2 2B teacher — 1024-d pixel encoder."""
try:  # package import
    from . import model, infer  # noqa: F401
except ImportError:  # standalone folder
    import model, infer  # noqa: F401

load_model = model.load_model
encode_tile = infer.encode_tile
encode_pixels = infer.encode_pixels
