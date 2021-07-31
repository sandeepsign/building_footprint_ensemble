from aerialDetection.inference import unet_inference
from PIL import Image
import torchvision.transforms as transforms
import torch
import os

demo_im = Image.open('./demo.PNG')
demo_im_out = './demo_out.png'
to_tensor = transforms.ToTensor()
to_image = transforms.ToPILImage()
demo_im = to_tensor(demo_im)
#strip transparency layer
demo_im = demo_im[0:3, :, :]

state_dict_path = './model_state/project.pth'

inference = unet_inference(state_dict_path)

out = inference.run_inference(demo_im)
out = (torch.lt(out[0, :, :],out[1, :, :])).float()
out = to_image(out)
out.save(fp = demo_im_out)