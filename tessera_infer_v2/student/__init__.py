"""TESSERA v2 pixel students — 128-d Matryoshka {16, 32, 64, 128} encoders."""
try:  # package import
    from . import model, infer, quantize  # noqa: F401
except ImportError:  # standalone folder
    import model, infer, quantize  # noqa: F401

load_model = model.load_model
encode_tile = infer.encode_tile
encode_pixels = infer.encode_pixels
quantize_int8 = quantize.quantize
