from typing import Any, Iterator, List, Union
import numpy as np
import torch
import detectron2.layers


def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):

    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"
    N = len(masks)
    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype=torch.uint8)
    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor
    device = boxes.device
    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape

    # The actual implementation split the input into chunks,
    # and paste them chunk by chunk.
    if device.type == "cpu":
        # CPU is most efficient when they are pasted one by one with skip_empty=True
        # so that it performs minimal number of operations.
        num_chunks = N
    else:
        # GPU benefits from parallelism for larger chunks, but may have memory issue
        num_chunks = int(np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
        assert (
            num_chunks <= N
        ), "Default GPU_MEM_LIMIT in mask_ops.py is too small; try increasing it"
    chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device=device, dtype=torch.float32
    )
    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty=device.type == "cpu"
        )
        img_masks[(inds,) + spatial_inds] = masks_chunk

    return img_masks


detectron2.layers.mask_ops.paste_masks_in_image.__code__ = paste_masks_in_image.__code__
print('detectron2.layers.mask_ops.paste_masks_in_image is patched.')


def BitMasks__init__(self, tensor: Union[torch.Tensor, np.ndarray]):

    device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
    assert tensor.dim() == 3, tensor.size()
    self.image_size = tensor.shape[1:]
    self.tensor = tensor

detectron2.structures.masks.BitMasks.__init__.__code__ = BitMasks__init__.__code__
print('detectron2.structures.masks.BitMasks.init is patched.')
