This project is based on Real-ESRGAN repo: https://github.com/xinntao/Real-ESRGAN

1. Put `train_SRResNet_x4_FFHQ_300k.yml` under the `options` folder.
2. Put `ffhqsub_dataset.py` under the `realesrgan/data` folder.


-> had to hardcode in 

as noted in https://github.com/XPixelGroup/BasicSR/issues/649

```from torchvision.transforms.functional import rgb_to_grayscale``` 