# Modified by RV-SDTM contributors for this public release; see NOTICE.
"""Dense encoder-decoder layers used by the range-view feature extractor."""

from functools import partial

import torch.nn as nn

from pcdet.models.model_utils.sparse_block_utils import (
    BasicBlock,
    post_act_block_dense_2d,
)


norm_fn_2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)


class DEDLayer(nn.Module):
    """Multi-level dense encoder-decoder block with additive skip fusion."""

    def __init__(self, dim: int, down_stride: list, num_SBB: list):
        super().__init__()

        self.encoder = nn.ModuleList(
            nn.Sequential(*[BasicBlock(dim) for _ in range(num_SBB[0])])
        )

        num_levels = len(down_stride)
        for idx in range(1, num_levels):
            cur_layers = [BasicBlock(dim, downsample=True)]
            cur_layers.extend(
                [BasicBlock(dim) for _ in range(num_SBB[idx])]
            )
            self.encoder.append(nn.Sequential(*cur_layers))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(
                post_act_block_dense_2d(
                    dim,
                    dim,
                    down_stride[idx],
                    down_stride[idx],
                    0,
                    conv_type='deconv',
                )
            )
            self.decoder_norm.append(norm_fn_2d(dim))

    def forward(self, x):
        features = []
        for encoder in self.encoder:
            x = encoder(x)
            features.append(x)

        for decoder, norm, skip in zip(
                self.decoder, self.decoder_norm, features[:-1][::-1]):
            x = norm(decoder(x) + skip)

        return x
