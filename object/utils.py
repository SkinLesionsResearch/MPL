import network

def get_model(net, num_classes):
    if net[0:3] == 'res':
        return network.ResBase(net, num_classes).cuda()
    elif net[0:3] == 'vgg':
        return network.VGGBase(net, num_classes).cuda()
    elif net[0:3] == 'inc':
        return network.InceptionBase(num_classes).cuda()
    elif net[0:3] == 'goo':
        return network.GoogLeNet(num_classes).cuda()
    elif net[0:3] == 'ale':
        return network.AlexNet(num_classes).cuda()