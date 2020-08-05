"""
Created on Tue Aug  4 20:26:23 2020

@author: medha iyer
"""


import torch
from torch import nn

class SimpleConv(nn.Module):
  def __init__(self, num_classes, spiral_size, meander_size, circle_size):
      super(SimpleConv, self).__init__()
      self.spiral_size = spiral_size
      self.meander_size = meander_size
      self.circle_size = circle_size
      self.dim = 0 #dimensions of image after concatenation (set in forward())

      self.spiral_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )

      self.meander_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )

      self.circle_nn = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(64, 192, kernel_size=5, padding=2),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(192, 384, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(384, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )

      self.concat_nn = nn.Sequential(
          nn.Dropout(),
          nn.Linear(self.dim, 4096),
          nn.ReLU(inplace=True),
          nn.Dropout(),
          nn.Linear(4096, 4096),
          nn.ReLU(inplace=True),
          nn.Linear(4096, num_classes),
      )

  def forward(self, spirals, meanders, circles):
      spirals = self.spiral_nn(spirals)
      spirals = spirals.view(spirals.size(0), -1)

      meanders = self.meander_nn(meanders)
      meanders = meanders.view(meanders.size(0), -1)

      circles = self.circle_nn(circles)
      circles = circles.view(circles.size(0), -1)

      # now we can concatenate them
      combined = torch.cat((spirals, meanders, circles), dim=1)
      self.dim = combined.shape()[1]
      out = self.concat_nn(combined)

      return out

# When we import as a module in another file, we can declare SimpleConv variable
# num_classes is the size of each input sample (how many samples are we breaking training data into?)
#   def simple_conv(pretrained=False, num_classes=2):
#       model = SimpleConv(num_classes, spiral_size=756*786, meander_size=744*822, circle_size=675*720)
#       return model

# dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)}

if __name__ =='__main__':
    num_classes = 2
    model = SimpleConv(num_classes, spiral_size=756*786, meander_size=744*822, circle_size=675*720)