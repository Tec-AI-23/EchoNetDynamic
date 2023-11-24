# For now, this are only meant to work on the /scripts folder.

ECHONET = "../EchoNet-Dynamic"
VIDEOS = "../EchoNet-Dynamic/videos"

DATA = "../EchoNet-Dynamic/data"

TRAIN = "../EchoNet-Dynamic/data/train"
VALIDATION = "../EchoNet-Dynamic/validation"

def split(s):
    if s == "train":
        IMAGES= "../EchoNet-Dynamic/data/train/images"
        MASKS = "../EchoNet-Dynamic/data/train/masks"
        HEATMAPS = "../EchoNet-Dynamic/data/train/heatmaps"
    elif s == "validation":
        IMAGES= "../EchoNet-Dynamic/data/validation/images"
        MASKS = "../EchoNet-Dynamic/data/validation/masks"
        HEATMAPS = "../EchoNet-Dynamic/data/validation/heatmaps"
    return IMAGES,MASKS,HEATMAPS

