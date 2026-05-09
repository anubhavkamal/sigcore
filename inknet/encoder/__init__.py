from .model import InkEncoder, InkEncoderThin, InkEncoderCompact

available_models = {
    'inkencoder': InkEncoder,
    'inkencoder_thin': InkEncoderThin,
    'inkencoder_compact': InkEncoderCompact,
}
