import torch.nn as nn
import torch.nn.functional as F
import torch


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, gelu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel,
                                             kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if gelu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class depthWiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(depthWiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            depthWiseConv(in_channel, out_channel),
            nn.GELU(),
            depthWiseConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.main(x) + x


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel)
                  for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DMFF(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.n_feat = n_feat
        self.conv1 = nn.Conv2d(n_feat * 2, n_feat * 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(n_feat, n_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, y):
        # fusion propagation
        feat_fusion = torch.cat([x, y], dim=1)  # b 128 256 256
        feat_fusion = self.lrelu(self.conv1(feat_fusion))  # b 128 256 256
        feat_prop1, feat_prop2 = torch.split(feat_fusion, self.n_feat, dim=1)
        feat_prop1 = feat_prop1 * torch.sigmoid(self.conv2(feat_prop1))
        feat_prop2 = feat_prop2 * torch.sigmoid(self.conv3(feat_prop2))
        x = feat_prop1 + feat_prop2
        return x

class LWDNet(nn.Module):
    def __init__(self, num_res=8, base_channel=32):
        super().__init__()

        # stage_1
        self.stage1_convIn = BasicConv(3, base_channel, kernel_size=3, gelu=True, stride=1)
        self.stage1_encoder = EBlock(base_channel, num_res)

        # stage_2
        self.stage2_SCM = nn.Conv2d(3, base_channel * 2, kernel_size=3, stride=1, padding=1)
        self.stage2_convIn = BasicConv(base_channel, base_channel * 2, kernel_size=3, gelu=True, stride=2)
        self.stage2_atb = DMFF(base_channel * 2)
        self.stage2_encoder = EBlock(base_channel * 2, num_res)

        # stage_3
        self.stage3_SCM = nn.Conv2d(3, base_channel * 4, kernel_size=3, stride=1, padding=1)
        self.stage3_convIn = BasicConv(base_channel * 2, base_channel * 4, kernel_size=3, gelu=True, stride=2)
        self.stage3_atb = DMFF(base_channel * 4)
        self.stage3_encoder = EBlock(base_channel * 4, num_res)

        # stage_3
        self.stage3_decoder = DBlock(base_channel * 4, num_res)
        self.stage3_transpose = BasicConv(base_channel * 4, base_channel * 2, kernel_size=4, gelu=True, stride=2,
                                          transpose=True)
        self.stage3_convOut = BasicConv(base_channel * 4, 3, kernel_size=3, gelu=False, stride=1)

        # stage_2
        self.stage2_feat_extract = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, gelu=True, stride=1)
        self.stage2_decoder = DBlock(base_channel * 2, num_res)
        self.stage2_transpose = BasicConv(base_channel * 2, base_channel, kernel_size=4, gelu=True, stride=2,
                                          transpose=True)
        self.stage2_convOut = BasicConv(base_channel * 2, 3, kernel_size=3, gelu=False, stride=1)

        # stage_1
        self.stage1_feat_extract = BasicConv(base_channel * 2, base_channel, kernel_size=1, gelu=True, stride=1)
        self.stage1_decoder = DBlock(base_channel, num_res)
        self.stage1_convOut = BasicConv(base_channel, 3, kernel_size=3, gelu=False, stride=1)

    def forward(self, x):
        outputs = list()
        '''
        b, c, h, w = x.shape
        padsize = 16
        h_n = (padsize - h % padsize) % padsize
        w_n = (padsize - w % padsize) % padsize
        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        '''
        stage2_x = F.interpolate(x, scale_factor=0.5)
        stage3_x = F.interpolate(stage2_x, scale_factor=0.5)
        stage2_z2 = self.stage2_SCM(stage2_x)
        stage3_z2 = self.stage3_SCM(stage3_x)

        # encoder
        stage1_x_shallow_feature = self.stage1_convIn(x)
        stage1_res = self.stage1_encoder(stage1_x_shallow_feature)

        stage2_z = self.stage2_convIn(stage1_res)
        stage2_z = self.stage2_atb(stage2_z, stage2_z2)
        stage2_res = self.stage2_encoder(stage2_z)

        stage3_z = self.stage3_convIn(stage2_res)
        stage3_z = self.stage3_atb(stage3_z, stage3_z2)
        stage3_res = self.stage3_encoder(stage3_z)

        # decoder
        stage3_out = self.stage3_decoder(stage3_res)
        stage3_out_ = self.stage3_transpose(stage3_out)
        stage3_out = self.stage3_convOut(stage3_out)
        outputs.append(stage3_x + stage3_out)

        stage2_out = torch.cat([stage3_out_, stage2_res], dim=1)
        stage2_out = self.stage2_feat_extract(stage2_out)
        stage2_out = self.stage2_decoder(stage2_out)
        stage2_out_ = self.stage2_transpose(stage2_out)
        stage2_out = self.stage2_convOut(stage2_out)
        outputs.append(stage2_out + stage2_x)

        stage1_out = torch.cat([stage2_out_, stage1_res], dim=1)
        stage1_out = self.stage1_feat_extract(stage1_out)
        stage1_out = self.stage1_decoder(stage1_out)
        stage1_out = self.stage1_convOut(stage1_out)
        res = stage1_out + x
        # res = res[:, :, :h, :w]
        outputs.append(res)

        return outputs


def get_params(net):
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    x = torch.randn(1, 3, 256, 256)
    print(x.shape)
    # model = NAFNet()
    # print(model)
    print(f'params: {sum(map(lambda x: x.numel(), net.parameters()))}')
    print(flop_count_table(FlopCountAnalysis(net, x), activations=ActivationCountAnalysis(net, x)))
    # output = model(x)
    # print(output.shape)


def get_measure_time(net, counts):
    from collections import OrderedDict
    net.cuda()
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    input_data = torch.randn(1, 3, 720, 1280).cuda()
    print("Warm up..")
    with torch.no_grad():
        for _ in range(50):
            __ = net(input_data)
    print("start time...")
    torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(counts):
            start.record()
            __ = net(input_data)
            end.record()
            torch.cuda.synchronize()
            test_results["runtime"].append(start.elapsed_time(end))
        print("Average Time:", sum(test_results["runtime"]) / len(test_results["runtime"]))


if __name__ == '__main__':
    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    # import torch
    # x = torch.randn(1, 3, 256, 256)
    # print(x.shape)
    # model = LWDNet( num_res=12, base_channel=32)
    # # print(model)
    # print(f'params: {sum(map(lambda x: x.numel(), model.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)))
    # output = model(x)
    # print(output.shape)

    # from thop import profile
    # from thop import clever_format
    # import torch
    #
    # model = LWDNet(num_res=12, base_channel=32)
    # input = torch.randn(1, 3, 256, 256)
    # macs, params = profile(model, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs)
    # print(params)

    # from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    #
    # x = torch.randn(1, 3, 256, 256)
    # print(x.shape)
    # net=LWDNet(num_res=12,base_channel=32)
    # # model = NAFNet()
    # # print(model)
    # print(f'params: {sum(map(lambda x: x.numel(), net.parameters()))}')
    # print(flop_count_table(FlopCountAnalysis(net, x), activations=ActivationCountAnalysis(net, x)))
    net = LWDNet(num_res=12, base_channel=32)
    get_params(net)
    get_measure_time(net, 100)
