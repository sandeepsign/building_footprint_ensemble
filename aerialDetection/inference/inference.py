from aerialDetection.models import ResUNet
import torch
import torchvision.transforms.functional as TF
import torch.nn as nn



class unet_inference():
    def __init__(self, path_to_state_dict, num_out_channels=2):
        state_dict = torch.load(path_to_state_dict, map_location = torch.device('cpu'))
        self.model = ResUNet(num_blocks=4)
        self.model.load_state_dict(state_dict)
        self.softmax = nn.Softmax(dim=0)
        self.num_out_channels = num_out_channels
        self.model.eval()

    def set_output_channels(self, num_out_channels):
        self.num_out_channels = num_out_channels

    def run_inference(self, input):
        """Runs inference on input data
        
        input: 3x224x224 Tensor (not normalized)
        
        Returns 2x224x224 tensor with softmax probabilities where 0 channel is
        prob(~building) and 1 channel is prob(building)."""
        assert(len(input.shape)==3)
        #transform
        with torch.no_grad():
            input = TF.normalize(input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input = input.unsqueeze(0)
            output = self.model.forward(input)
            output = output.squeeze()
            output = self.softmax(output)
            if self.num_out_channels == 1:
                output = (torch.lt(output[0, :, :],output[1, :, :])).float()
            return output
