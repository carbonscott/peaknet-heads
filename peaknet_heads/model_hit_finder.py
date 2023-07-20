import torch
import torch.nn as nn

from peaknet.att_unet import AttentionUNet    # The trunk model

from .utils import TorchModelAttributeParser, NNSize
from functools import reduce


class TrunkNet(AttentionUNet):
    def __init__(self, base_channels,
                       in_channels,
                       out_channels,
                       uses_skip_connection,
                       att_gate_channels, ):
        super().__init__( base_channels        = base_channels,
                          in_channels          = in_channels,
                          out_channels         = out_channels,
                          uses_skip_connection = uses_skip_connection,
                          att_gate_channels    = None, )

        return None


    def forward(self, x):
        depth = self.depth

        # ___/ FEATURE EXTRACTION PATH \___
        fmap_fext_list = []
        for i in range(depth):
            double_conv = self.module_list_double_conv_fext[i]
            pool        = self.module_list_pool_fext[i]

            fmap = double_conv(x)
            fmap_fext_list.append(fmap)

            x = pool(fmap)

        double_conv = self.module_list_double_conv_fbot[0]    # Single element though, may not be a good design.
        x_now = double_conv(x)

        return x_now




class HitFinder(nn.Module):

    def __init__(self, model_trunk, C, H, W):
        '''
        C, H, W are the batch size, height and width of the feature maps.
        '''
        super().__init__()

        self.model_trunk = model_trunk

        # Freeze the parameters in the pf model...
        for param in self.model_trunk.parameters():
            param.requires_grad = False

        # Create a new prediction head...
        # ...Conv 1x1
        conv1x1 = nn.Sequential(
            # Conv 1x1 to combine feature maps...
            nn.Conv2d( in_channels  = C,
                       out_channels = 1,
                       kernel_size  = 1,
                       stride       = 1,
                       padding      = 0,
                       bias         = False, ),
            nn.BatchNorm2d( num_features = 1 ),
            nn.ReLU(),
        )

        # Fetch all input arguments that define the layer...
        attr_parser = TorchModelAttributeParser()
        conv_dict = {}
        for layer_name, model in conv1x1.named_children():
            conv_dict[layer_name] = attr_parser.parse(model)

        # Calculate the output size...
        self.in_features = reduce(lambda x, y: x * y, NNSize(H, W, C, conv_dict).shape())

        # ...MLP
        mlp = nn.Sequential(
            nn.Linear( in_features  = self.in_features,
                       out_features = 128,
                       bias = True),
            nn.ReLU(),
            nn.Linear( in_features  = 128,
                       out_features = 1,
                       bias = True),
        )

        self.classifier = nn.ModuleDict( {"conv" : conv1x1, "mlp" : mlp} )


    def forward(self, x):
        x = self.model_trunk(x)

        x = self.classifier["conv"](x)
        x = x.view(-1, self.in_features)
        x = self.classifier["mlp"](x)

        return x
