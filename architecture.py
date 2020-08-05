"""
Created on Tue Aug  4 20:26:23 2020

@author: medha iyer
"""

# #import keras
# from keras.models import Sequential
# from keras.layers import Dense
#
# # define three sets of inputs
# spiral_test = Input()
# meander_test = Input()
# circle_test = Input()
#

import torch
from torch import nn

class TwoInputsNet(nn.Module):
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    self.conv = nn.Conv2d( ... )  # set up your layer here
    self.fc1 = nn.Linear( ... )  # set up first FC layer
    self.fc2 = nn.Linear( ... )  # set up the other FC layer

  def forward(self, input1, input2):
    c = self.conv(input1)
    f = self.fc1(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out

  class SimpleConv(nn.Module):
      def __init__(self, num_classes):
          super(SimpleConv, self).__init__()
          self.features = nn.Sequential(
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
          self.classifier = nn.Sequential(
              nn.Dropout(),
              nn.Linear(256 * 6 * 6, 4096),
              nn.ReLU(inplace=True),
              nn.Dropout(),
              nn.Linear(4096, 4096),
              nn.ReLU(inplace=True),
              nn.Linear(4096, num_classes),
          )

      def forward(self, spirals, meanders, circles):
          spirals = self.features(spirals)
          spirals = spirals.view(spirals.size(0), 256 * 6 * 6) #TODO find what this line does
          spirals = self.classifier(spirals)

          meanders = self.features(meanders)
          meanders = meanders.view(meanders.size(0), 256 * 6 * 6)
          meanders = self.classifier(meanders)

          circles = self.features(circles)
          circles = circles.view(circles.size(0), 256 * 6 * 6)
          circles = self.classifier(circles)

          # now we can reshape these to 2D and concat them
          # TODO find what view we need to reshape to based on how we want it to output
          combined = torch.cat((c.view(c.size(0), -1),
                                f.view(f.size(0), -1)), dim=1)
          out = self.fc2(combined)

          return out

# When we import as a module in another file, we can declare SimpleConv variable
# num_classes is the size of each input sample (how many samples are we breaking training data into?)
  def simple_conv(pretrained=False, num_classes=140):
      model = SimpleConv(num_classes)
      # if pretrained:
      #     model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
      return model