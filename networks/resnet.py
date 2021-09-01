import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50,
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}
# testing
class ResBase(nn.Module):
    def __init__(self, name, num_classes):
        super(ResBase, self).__init__()
        model_resnet = res_dict[name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        # self.fc = nn.Linear(model_resnet.fc.in_features, 1024)
        self.classifier = nn.Linear(model_resnet.fc.in_features, num_classes)

        self.fc_a = nn.Linear(model_resnet.fc.in_features, 384)
        self.fc_b = nn.Linear(model_resnet.fc.in_features, 384)

        self.fc_weight = nn.Linear(384, 1)
        self.fc_weight_sigmoid = nn.Sigmoid()


    def forward(self, x, afm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        # features = self.fc(x)
        logits = self.classifier(features)
        if afm:
            bs = int(features.shape[0] / 2)
            x_part1 = features[:bs, :]
            x_part2 = features[bs:, :]

            fc_a = F.relu(self.fc_a(x_part1), inplace=True)
            fc_b = F.relu(self.fc_b(x_part2), inplace=True)

            fc_weight = self.fc_weight(fc_a + fc_b)
            weight = self.fc_weight_sigmoid(fc_weight)

            mixup_x = weight * x_part1 + (1 - weight) * x_part2
            mixup_x = torch.cat([mixup_x, mixup_x], dim=0)

            afm_logits = self.classifier(mixup_x)
            return logits, afm_logits
        return features, logits

if __name__ == '__main__':
    import torch
    model = ResBase('resnet50', 2).cuda()
    rand = torch.rand(1, 3, 256, 256).cuda()
    cam, _, _ = model(rand)
    print(cam.shape)


