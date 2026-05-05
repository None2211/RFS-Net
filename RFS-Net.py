import torch
import torch.nn as nn
import torch.nn.functional as F
from CSwin import cswin_small




def dwt_haar(x):

    x1 = x[:, :, 0::2, :]  
    x2 = x[:, :, 1::2, :]   

    L = (x1 + x2) / 2.0   
    H = (x1 - x2) / 2.0   

    LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) / 2.0
    LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) / 2.0
    HL = (H[:, :, :, 0::2] + H[:, :, :, 1::2]) / 2.0
    HH = (H[:, :, :, 0::2] - H[:, :, :, 1::2]) / 2.0

    return LL, LH, HL, HH


def idwt_haar(LL, LH, HL, HH):

    B, C, h, w = LL.shape
    H, W = h * 2, w * 2


    L = torch.zeros(B, C, h, W, device=LL.device, dtype=LL.dtype)
    L[:, :, :, 0::2] = LL + LH
    L[:, :, :, 1::2] = LL - LH

    Hi = torch.zeros(B, C, h, W, device=LL.device, dtype=LL.dtype)
    Hi[:, :, :, 0::2] = HL + HH
    Hi[:, :, :, 1::2] = HL - HH


    out = torch.zeros(B, C, H, W, device=LL.device, dtype=LL.dtype)
    out[:, :, 0::2, :] = L + Hi
    out[:, :, 1::2, :] = L - Hi

    return out




class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)




class decoderbox(nn.Module):  # n,c,h,w -> n,c/2,2h,2w
    def __init__(self, in_planes, out_planes):
        super(decoderbox, self).__init__()
        self.eca = eca_layer(channel=in_planes)
        self.act = nn.GELU()

        self.conv1 = nn.Conv2d(
            in_planes, in_planes // 4,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm1 = nn.LayerNorm(in_planes // 4, eps=1e-6)

        self.deconv = nn.ConvTranspose2d(
            in_planes // 4, in_planes // 4,
            3, stride=2, padding=1, output_padding=1
        )
        self.norm2 = nn.LayerNorm(in_planes // 4, eps=1e-6)

        self.conv2 = nn.Conv2d(
            in_planes // 4, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm3 = nn.LayerNorm(out_planes, eps=1e-6)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.eca(x)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        x = self.deconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm3(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)

        return x


class outconv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(outconv, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_planes, out_planes,
            3, stride=2, padding=1, output_padding=1
        )
        self.norm = nn.LayerNorm(out_planes, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        return x




class RSWML(nn.Module):


    def __init__(self, in_channels, init_alpha=0.2):
        super().__init__()
        self.in_channels = in_channels

        def make_band_conv(c):

            return nn.Sequential(
                nn.Conv2d(c, c, kernel_size=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=1, bias=False)
            )

        def make_gate(c):

            return nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, kernel_size=1, bias=False),
                nn.Sigmoid()
            )

        # LL / LH / HL / HH
        self.band_convs = nn.ModuleList([make_band_conv(in_channels) for _ in range(4)])
        self.spatial_gates = nn.ModuleList([make_gate(in_channels) for _ in range(4)])


        self.band_logits = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.band_alpha  = nn.Parameter(torch.full((4,), float(init_alpha), dtype=torch.float32))

    def _process_band(self, subband, band_conv, gate_net):

        feat = band_conv(subband)      
        gate = gate_net(feat)           
        return gate

    @torch.no_grad()
    def get_band_params(self):
        mix_w = torch.softmax(self.band_logits, dim=0).detach().cpu().numpy()
        alpha = F.softplus(self.band_alpha).detach().cpu().numpy()
        return {
            "w_LL": float(mix_w[0]), "w_LH": float(mix_w[1]),
            "w_HL": float(mix_w[2]), "w_HH": float(mix_w[3]),
            "a_LL": float(alpha[0]), "a_LH": float(alpha[1]),
            "a_HL": float(alpha[2]), "a_HH": float(alpha[3]),
        }

    def forward(self, x):
        B, C, H, W = x.shape


        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')


        LL, LH, HL, HH = dwt_haar(x)
        subbands = [LL, LH, HL, HH]


        gates_full = []
        for i, sb in enumerate(subbands):
            gate_sub = self._process_band(sb, self.band_convs[i], self.spatial_gates[i])

            gate_full = F.interpolate(
                gate_sub, size=(x.shape[2], x.shape[3]),
                mode='bilinear', align_corners=False
            )
            gates_full.append(gate_full)

        mix_w = torch.softmax(self.band_logits, dim=0)         
        alpha  = F.softplus(self.band_alpha)                   

        delta = 0.0
        for i in range(4):
            delta = delta + mix_w[i] * alpha[i] * (x * gates_full[i])

        out = x + delta


        out = out[:, :, :H, :W]
        return out



class DMFSIL(nn.Module):
    def __init__(self, dim, reduction=8):
        super(DMFSIL, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim * 2, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_trans, x_sp):
        B, C, H, W = x_trans.shape

        if x_sp.shape[2:] != x_trans.shape[2:]:
            x_sp = F.interpolate(x_sp, size=(H, W), mode='bilinear', align_corners=False)

        x_sum = x_trans + x_sp
        s = self.avg_pool(x_sum).view(B, C)
        weights = self.fc(s).view(B, 2, C)
        weights = self.softmax(weights)

        w_trans = weights[:, 0, :].view(B, C, 1, 1)
        w_sp    = weights[:, 1, :].view(B, C, 1, 1)

        return x_trans * w_trans + x_sp * w_sp



class FSGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FSGB, self).__init__()

        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.structure_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
        feat_main = self.main_conv(x)
        gate = self.structure_conv(x)
        out = feat_main + (feat_main * gate) * self.alpha
        out = self.out_conv(out)
        return out



class RSWM(nn.Module):
    def __init__(self):
        super(RSWM, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.spec1 = RSWML(in_channels=64,  init_alpha=0.15)

        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.spec2 = RSWML(in_channels=128, init_alpha=0.15)

        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.spec3 = RSWML(in_channels=256, init_alpha=0.10)

        self.down3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.spec4 = RSWML(in_channels=512, init_alpha=0.08)

    def forward(self, x):
        f1 = self.stem(x)
        f1 = self.spec1(f1)

        f2 = self.down1(f1)
        f2 = self.spec2(f2)

        f3 = self.down2(f2)
        f3 = self.spec3(f3)

        f4 = self.down3(f3)
        f4 = self.spec4(f4)

        return f1, f2, f3, f4



class RFS-Net(nn.Module):
    def __init__(self, num_class, num_cls):
        super(RFS-Net, self).__init__()

        self.backbone = cswin_small()
        path = 'cswin_small_224.pth'
        save_model = torch.load(path, map_location='cpu')
        model_dict = self.backbone.state_dict()
        state_dict = {
            k: v for k, v in save_model['state_dict_ema'].items()
            if k in model_dict.keys()
        }
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.sp_injector = RSWM()

        self.fusion1 = DMFSIL(dim=64)
        self.fusion2 = DMFSIL(dim=128)
        self.fusion3 = DMFSIL(dim=256)
        self.fusion4 = DMFSIL(dim=512)

        self.mix = nn.Parameter(torch.FloatTensor(7))
        self.mix.data.fill_(1.0)

        self.up5 = decoderbox(512, 256)
        self.up4 = decoderbox(256, 128)
        self.up3 = decoderbox(128, 64)
        self.up2 = decoderbox(64, 64)

        self.gate5 = FSGB(256, 256)
        self.gate4 = FSGB(128, 128)
        self.gate3 = FSGB(64, 64)
        self.gate2 = FSGB(64, 64)

        self.outconv = outconv(64, num_class)
        self.logit1 = nn.Conv2d(64,       num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(64,       num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128,      num_class, kernel_size=1)
        self.logit0 = nn.Conv2d(num_class, num_class, kernel_size=1)
        self.logit5 = nn.Conv2d(512,      num_class, kernel_size=1)
        self.logit6 = nn.Conv2d(256,      num_class, kernel_size=1)

        self.num_class = num_class
        self.num_cls   = num_cls
        self.cls_head  = nn.Linear(256, num_cls)

    def forward(self, x, superpixel):
        _, _, H, W = x.shape

        sp1, sp2, sp3, sp4 = self.sp_injector(superpixel)
        t1,  t2,  t3,  t4  = self.backbone(x)

        e1 = self.fusion1(t1, sp1)
        e2 = self.fusion2(t2, sp2)
        e3 = self.fusion3(t3, sp3)
        e4 = self.fusion4(t4, sp4)

        e5 = e4

        up5 = self.up5(e5)
        up5 = up5 + e3
        up5 = self.gate5(up5)

        up4 = self.up4(up5)
        up4 = up4 + e2
        up4 = self.gate4(up4)

        up3 = self.up3(up4)
        up3 = up3 + e1
        up3 = self.gate3(up3)

        up2 = self.up2(up3)
        up2 = self.gate2(up2)

        out = self.outconv(up2)

        logit1 = F.interpolate(self.logit1(up2), size=(H, W), mode='bilinear', align_corners=False)
        logit2 = F.interpolate(self.logit2(up3), size=(H, W), mode='bilinear', align_corners=False)
        logit3 = F.interpolate(self.logit3(up4), size=(H, W), mode='bilinear', align_corners=False)
        logit0 = F.interpolate(self.logit0(out), size=(H, W), mode='bilinear', align_corners=False)
        logit5 = F.interpolate(self.logit5(e5),  size=(H, W), mode='bilinear', align_corners=False)
        logit6 = F.interpolate(self.logit6(up5), size=(H, W), mode='bilinear', align_corners=False)

        logit = (
            self.mix[1] * logit1 +
            self.mix[2] * logit2 +
            self.mix[3] * logit3 +
            self.mix[4] * logit0 +
            self.mix[5] * logit5 +
            self.mix[6] * logit6
        )

        return logit


if __name__ == "__main__":
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_img = torch.randn(1, 3, 224, 224).to(device)
    sp_img    = torch.randn(1, 3, 224, 224).to(device)

    model = RFS-Net(num_cls=1, num_class=1).to(device)
    flops, params = profile(model, inputs=(input_img, sp_img))

    print(f"Total Parameters: {params / 1e6:.2f} M")
    print(f"Total GFLOPs: {flops / 1e9:.3f} G")
