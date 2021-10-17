import albumentations as A


def get_transforms(**transform_args):

    train_transforms = A.Compose([
        A.Resize(*transform_args['resize_shape'], interpolation=4, always_apply=True, p=1),
        A.HorizontalFlip(p=transform_args['horizontal_flip_probability']),
        A.VerticalFlip(p=transform_args['vertical_flip_probability']),
    ])

    test_transforms = A.Compose([
        A.Resize(*transform_args['resize_shape'], interpolation=4, always_apply=True, p=1),
    ])

    return {'train': train_transforms, 'test': test_transforms}
