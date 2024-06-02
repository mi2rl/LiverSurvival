import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Second(nn.Module):

    def __init__(self,
                 num_init_features=15,
                 num_classes=1000, norm='bn', nb_cat=6, chn=128, ft=False):

        super(Second, self).__init__()

        self.ft=ft
        self.nb_cat = nb_cat

        num_features = num_init_features

        # Linear layer

        self.sigmoid = torch.nn.Sigmoid()

        self.classifier_th3 = nn.Sequential(nn.Linear(num_features+self.nb_cat, chn))
        self.classifier_th3_2 = nn.Sequential(nn.Linear(num_features+20, chn))
        self.classifier_th4 = nn.Sequential(nn.Linear(chn, num_classes))

    def forward(self, x, y):
        if self.ft==False:

            out = torch.cat([x,y], dim=1)
            out = self.classifier_th3(out)
            out = self.classifier_th4(out)

        elif self.ft==True:
            y1,y2 = y
            out = torch.cat([x,y1,y2], dim=1)
            out = self.classifier_th3_2(out)
            out = self.classifier_th4(out)

        out = self.sigmoid(out)

        return out