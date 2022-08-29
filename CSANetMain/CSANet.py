import torch.nn as nn
import torch.nn.functional as F
from CSANetMain import attontion
import torch




class deeplab_V2(nn.Module):
    def __init__(self):
        super(deeplab_V2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=198, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        '''
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),#![](classification_maps/IN_gt.png)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),

        )  #进行卷积操作


        inter_channels = 512 // 4###################################################
        self.conv5a = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.sa = attontion.PAM_Module(inter_channels)####
        self.sc = attontion.CAM_Module(inter_channels)
        self.sco = attontion.CoAM_Module(inter_channels)###
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))

        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=6, padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=12, padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        # self.fc8 = nn.Softmax2d()
        # self.fc8 = fun.l2normalization(scale=1)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        feat11 = self.conv5a(x1)
        sa_feat1 = self.sa(feat11)
        sa_conv1 = self.conv51(sa_feat1)
        sa_output1 = self.conv6(sa_conv1)

        feat12 = self.conv5c(x1)
        sc_feat1 = self.sc(feat12)
        sc_conv1 = self.conv52(sc_feat1)
        sc_output = self.conv7(sc_conv1)

        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        feat21 = self.conv5a(x2)
        sa_feat2 = self.sa(feat21)
        sa_conv2 = self.conv51(sa_feat2)
        sa_output2 = self.conv6(sa_conv2)

        feat22 = self.conv5c(x2)
        sc_feat2 = self.sc(feat22)
        sc_conv2 = self.conv52(sc_feat2)
        sc_output = self.conv7(sc_conv2)

        sco_conv1 = self.sco(feat11,feat12)
        sco_conv1 = self.conv51(sco_conv1)
        sco_conv2 = self.sco(feat12,feat11)
        sco_conv2 = self.conv51(sco_conv2)

        feat_sum1 = sa_conv1 + sc_conv1 + 0.3*sco_conv1
        feat_sum2 = sa_conv2 + sc_conv2 + 0.3*sco_conv2
        sasc_output1 = self.conv8(feat_sum1)
        sasc_output2 = self.conv8(feat_sum2)


        return sasc_output1, sasc_output2


class SiameseNet(nn.Module):
    def __init__(self, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.CNN = deeplab_V2()
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128, 1))


        if norm_flag == 'l2':
            self.norm = F.normalize  #F.normalize对输入的数据（tensor）进行指定维度的L2_norm运算
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):
        t0 = t0.float()
        t1 = t1.float()
        out_t0_embedding, out_t1_embedding, = self.CNN(t0, t1)
        return out_t0_embedding, out_t1_embedding


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect', bias=False)  # [64, 24, 24]
        self.bat1 = nn.BatchNorm2d(64)#
        self.reli1 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat2 = nn.BatchNorm2d(32)
        self.reli2 = nn.LeakyReLU(0.2)

        self.conv3 = nn.Conv2d(64, 256, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat3 = nn.BatchNorm2d(256)
        self.reli3 = nn.LeakyReLU(0.2)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):
        con1 = self.conv1(x)
        ba1 = self.bat1(con1)
        re1 = self.reli1(ba1)
        po1 = self.pool1(re1)
        con2 = self.conv2(po1)
        ba2 = self.bat2(con2)
        re2 = self.reli2(ba2)

        return re2


class ChangeNet(nn.Module):
    def __init__(self):
        super(ChangeNet, self).__init__()
        self.singlebrach = Classifier()# re2
        self.fc = nn.Sequential(        #一个有序的容器
            nn.Linear(32, 16),#32和16是维度
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, t0, t1):

        indata = t0 - t1
        c3 = self.singlebrach(indata)

        return c3


class Finalmodel(nn.Module):#######################nn,Module################################################################
    def __init__(self):
        super(Finalmodel, self).__init__()
        self.siamesnet = SiameseNet()
        self.chnet = ChangeNet() #c3
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, t0, t1):
        t0 = t0.permute(0, 3, 1, 2) #换个顺序 0123-----0312
        t1 = t1.permute(0, 3, 1, 2)

        x1, x2 = self.siamesnet(t0, t1)
        out = self.chnet(x1, x2)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.sigmoid(out)


        return out
