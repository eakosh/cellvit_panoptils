import albumentations as A


def get_train_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        # Geometric 
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Downscale(
            scale_min=0.5,
            scale_max=0.5,
            p=0.15,
        ),
        A.RandomSizedCrop(
            min_max_height=(int(image_size * 0.5), image_size),
            height=image_size,
            width=image_size,
            p=0.1,
        ),
        A.ElasticTransform(
            alpha=0.5,
            sigma=25,
            alpha_affine=15,
            border_mode=0,
            p=0.2,
        ),

        # Intensity / colour 
        A.Blur(blur_limit=10, p=0.2),
        A.GaussNoise(var_limit=(0.0, 50.0), p=0.25),
        A.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.1,
            hue=0.05,
            p=0.2,
        ),
        A.Superpixels(
            p_replace=0.1,
            n_segments=200,
            max_size=image_size // 2,
            p=0.1,
        ),
        A.ZoomBlur(max_factor=(1.0, 1.05), p=0.1),

        # Normalization  
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),

    ], additional_targets={
        'mask0': 'mask',  # tissue_mask
        'mask1': 'mask',  # instance_map
        'mask2': 'mask',  # nuclei_type_map
    })


def get_val_transforms(image_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ], additional_targets={
        'mask0': 'mask',
        'mask1': 'mask',
        'mask2': 'mask',
    })


class AlbumentationsWrapper:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, image, masks):
        if len(masks) != 3:
            raise ValueError(f"Expected 3 masks, got {len(masks)}")

        augmented = self.transforms(
            image=image,
            mask0=masks[0],
            mask1=masks[1],
            mask2=masks[2],
        )

        return {
            'image': augmented['image'],
            'masks': [
                augmented['mask0'],
                augmented['mask1'],
                augmented['mask2'],
            ],
        }


def create_train_transforms(image_size: int = 256) -> AlbumentationsWrapper:
    return AlbumentationsWrapper(get_train_transforms(image_size))


def create_val_transforms(image_size: int = 256) -> AlbumentationsWrapper:
    return AlbumentationsWrapper(get_val_transforms(image_size))
