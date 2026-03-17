import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn import init
import numbers
from einops import rearrange
from utils.brightness_align import (
    gamma_linearize_tensor,
    gamma_encode_tensor,
    align_to_brightest_tensor
)
from torch.utils.checkpoint import checkpoint

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class Restormer(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=60, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False,        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        use_checkpoint=False
    ):

        super(Restormer, self).__init__()

        #self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.use_checkpoint = use_checkpoint

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        def _checkpoint_wrapper(module, *args):
            if self.use_checkpoint and self.training:
                # use_reentrant=False 是 PyTorch 推荐的更稳定的写法
                return checkpoint(module, *args, use_reentrant=False)
            return module(*args)
        #inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1=inp_img
        out_enc_level1 = _checkpoint_wrapper(self.encoder_level1, inp_enc_level1)
        # _checkpoint_wrapper(self.encoder_level1, inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = _checkpoint_wrapper(self.encoder_level2, inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = _checkpoint_wrapper(self.encoder_level3, inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = _checkpoint_wrapper(self.latent, inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = _checkpoint_wrapper(self.decoder_level3, inp_dec_level3) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = _checkpoint_wrapper(self.decoder_level2, inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = _checkpoint_wrapper(self.decoder_level1, inp_dec_level1)
        
        out_dec_level1 = self.refinement(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1
    
def make_layer(block, num_blocks, **kwarg):
    """构建由相同块堆叠而成的网络层
    
    Args:
        block (nn.Module): 基础块类
        num_blocks (int): 堆叠块的数量
        **kwarg: 传递给块类的关键字参数
        
    Returns:
        nn.Sequential: 按顺序堆叠的块结构
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)

def flow_warp(x, flow, interpolation='bilinear', padding_mode='zeros', align_corners=True):
    """使用光流对图像或特征图进行变形
    
    Args:
        x (Tensor): 输入张量 (n, c, h, w)
        flow (Tensor): 光流张量 (n, h, w, 2)，最后一个维度是宽高方向的偏移量
        interpolation (str): 插值方式，默认双线性
        padding_mode (str): 填充方式，默认零填充
        align_corners (bool): 是否对齐角点，默认True
        
    Returns:
        Tensor: 变形后的图像或特征图
    """
    # 验证输入尺寸
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f'输入({x.size()[-2:]})和光流({flow.size()[1:3]})空间尺寸不匹配')
    
    _, _, h, w = x.size()
    device = flow.device
    
    # 创建网格坐标
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=device, dtype=x.dtype),
        torch.arange(0, w, device=device, dtype=x.dtype))
    grid = torch.stack((grid_x, grid_y), 2)  # (h, w, 2)
    grid.requires_grad = False

    # 应用光流偏移
    grid_flow = grid + flow
    # 归一化到[-1,1]范围
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
    
    # 使用grid_sample进行变形
    return F.grid_sample(
        x, grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)

def load_spynet(net, path):
    """加载预训练的SPyNet模型权重
    
    Args:
        net (nn.Module): 要加载权重的网络
        path (str): 权重文件路径
    """
    if isinstance(net, torch.nn.DataParallel):
        net = net.module
    state_dict = torch.load(path)

    print(f'从 {path} 加载模型')
    if hasattr(state_dict, '_metadata'):
        del state_dict._metadata  # 删除元数据（如果有）

    net_state = net.state_dict()
    is_loaded = {n:False for n in net_state.keys()}
    
    # 权重加载和名称映射
    for name, param in state_dict.items():
        # 替换旧版本参数名称
        name = name.replace('basic_module.0.conv', 'basic_module.0')
        name = name.replace('basic_module.1.conv', 'basic_module.2')
        name = name.replace('basic_module.2.conv', 'basic_module.4')
        name = name.replace('basic_module.3.conv', 'basic_module.6')
        name = name.replace('basic_module.4.conv', 'basic_module.8')
        
        # if name in net_state:
        #     try:
        #         net_state[name].copy_(param)  # 复制参数
        #         is_loaded[name] = True
        #     except Exception:
        #         # 处理尺寸不匹配错误
        #         print(f'参数 {name} 维度不匹配: 模型维度 {list(net_state[name].shape)}, 检查点维度 {list(param.shape)}')
        #         raise RuntimeError
        # else:
        #     print(f'跳过未使用参数 {name}')



        if name in net_state:
            # --- 关键修改：处理维度不匹配的 feat_extract 第一层 ---
            if name == 'align.feat_extract.main.0.weight' and param.shape != net_state[name].shape:
                # 原权重是 [out, 8, 3, 3]，我们需要 [out, 6, 3, 3]
                # 假设前 4 通道是原始 RAW(这里对应你的 JPG)，后 4 通道是校正流
                # 我们各取前 3 通道拼接，或者直接取前 6 通道
                print(f'正在对 {name} 进行通道裁剪: {list(param.shape)} -> {list(net_state[name].shape)}')
                
                # 方案：假设原 8 通道是 [R,Gr,Gb,B, R',Gr',Gb',B']
                # 你的 6 通道是 [R,G,B, R',G',B']
                # 我们可以手动选择索引 [0,1,2, 4,5,6] 来保留 RGB 对应的权重
                cropped_param = param[:, [0, 1, 2, 4, 5, 6], :, :] 
                net_state[name].copy_(cropped_param)
                is_loaded[name] = True
                continue
            
            # --- 关键修改：处理 conv_first 的帧数不匹配 ---
            if name == 'conv_first.weight' and param.shape != net_state[name].shape:
                # 原权重是 [64*9, dim, 4, 4]，我们需要 [64*7, dim, 4, 4]
                # 我们可以取前 7 帧的权重
                print(f'正在对 {name} 进行帧数裁剪: {list(param.shape)} -> {list(net_state[name].shape)}')
                cropped_param = param[:64*7, :, :, :] 
                net_state[name].copy_(cropped_param)
                is_loaded[name] = True
                continue

            else:
                try:
                    net_state[name].copy_(param)  # 复制参数
                    is_loaded[name] = True
                except Exception:
                    # 处理尺寸不匹配错误
                    print(f'参数 {name} 维度不匹配: 模型维度 {list(net_state[name].shape)}, 检查点维度 {list(param.shape)}')
                    raise RuntimeError
        else:
            print(f'跳过未使用参数 {name}')


    
    # 检查所有参数是否加载成功
    if all(is_loaded.values()):
        print(f'所有参数成功从 {path} 加载')
    else:
        for name in is_loaded:
            if not is_loaded[name]:
                print(f'参数 {name} 未初始化')

def init_weights(net, init_type='normal', init_gain=0.02):
    """初始化网络权重
    
    Args:
        net (nn.Module): 要初始化的网络
        init_type (str): 初始化类型（normal/xavier/kaiming等）
        init_gain (float): 初始化增益系数
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            # 根据类型选择初始化方法
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            elif init_type == 'uniform':
                init.uniform_(m.weight.data, b=init_gain)
            elif init_type == 'constant':
                init.constant_(m.weight.data, 0.0)
            else:
                raise NotImplementedError(f'不支持的初始化类型 {init_type}')
        elif hasattr(m, 'bias') and m.bias is not None:  # 初始化偏置
            init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:  # 批归一化层初始化
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)  # 递归应用初始化函数

class TMRNet(nn.Module):
    """时间多分辨率重建网络
    
    Args:
        opt: 配置参数
        mid_channels (int): 中间特征通道数，默认32
        max_residue_magnitude (int): 最大偏移幅度，默认10
    """
    def __init__(self, opt, mid_channels=64, max_residue_magnitude=10, use_checkpoint=False):
        super().__init__()
        self.mid_channels = mid_channels
        
        # 光流估计网络
        self.spynet = SPyNet()
        if opt.isTrain:  # 训练时加载预训练权重
            load_spynet(self.spynet, './model_zoo/06_spynet.pth')
        
        # 可变形对齐模块
        self.dcn_alignment = DeformableAlignment(
            mid_channels, mid_channels, 3, padding=1, 
            deform_groups=8, max_residue_magnitude=max_residue_magnitude, use_checkpoint=True)
        
        # 特征提取模块（包含输入卷积和残差块）
        self.feat_extract = ResidualBlocksWithInputConv(2*3, mid_channels, 5)#2*4
        self.use_checkpoint=use_checkpoint

    def compute_flow(self, lqs):
        """计算所有帧相对于第 4 帧 (index 3) 的光流
        
        Args:
            lqs (Tensor): 输入序列 (n, 7, c, h, w)
                
        Returns:
            Tuple[Tensor, Tensor]: 前向和后向光流 (n, 7, 2, h, w)
        """
        n, t, c, h, w = lqs.size()

        lqs_for_flow = torch.pow(torch.clamp(lqs, 0, 1), 1/2.2) 

        # 1. 提取并准备参考帧 (第 4 帧, index 3)
        # 形状变换: (n, 1, c, h, w) -> (n, 7, c, h, w) -> (n*7, c, h, w)
        ref = lqs_for_flow[:, 3:4, :, :, :].repeat(1, t, 1, 1, 1).view(-1, c, h, w)
        
        # 2. 准备所有待对齐帧
        oth = lqs_for_flow.view(-1, c, h, w)
        
        # 3. 计算反向光流 (从 oth 到 ref)
        # SPyNet 输出形状 (n*7, 2, h, w)
        flows_backward = self.spynet(ref, oth).view(n, t, 2, h, w)
        
        # 统一使用后向光流进行对齐
        return flows_backward, flows_backward


    def forward(self, lqs):
        """前向传播
        
        Args:
            lqs (Tensor): 输入序列 (n, 7, 3, h, w) - JPG 格式
                
        Returns:
            Tensor: 对齐后的特征序列 (n, 7, mid_channels, h, w)
        """
        def _checkpoint_wrapper(module, *args):
            if self.use_checkpoint and self.training:
                return checkpoint(module, *args, use_reentrant=False)
            return module(*args)
        n, t, c, h, w = lqs.size() # t=7, c=3
        
        # --- 步骤 1: 亮度对齐到最亮帧 ---
        # 1.1 将每帧分离为独立 Tensor 列表
        lqs_list = [lqs[:, i, :, :, :] for i in range(t)]  # [(n, 3, h, w), ...]

        # 1.2 亮度对齐到最亮帧 (使用 alig 的方法)
        aligned_list = align_to_brightest_tensor(lqs_list, gamma=2.2)

        # 1.3 构造 6 通道输入 (原始帧 + 亮度对齐帧)
        lqs_view = lqs.view(-1, c, h, w)  # 原始帧
        aligned_stack = torch.stack(aligned_list, dim=1).view(-1, c, h, w)  # 对齐后的帧
        lqs_in = torch.cat([lqs_view, aligned_stack], dim=1)  # (n*7, 6, h, w)

        # --- 步骤 2: 特征提取 ---
        feats_ = self.feat_extract(lqs_in) 
        h_f, w_f = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h_f, w_f) # (n, 7, mid_c, h_f, w_f)

        # --- 步骤 3: 特征对齐 ---
        # 3.1 获取光流（使用亮度对齐后的图像）
        aligned_stack_for_flow = torch.stack(aligned_list, dim=1)  # (n, t, 3, h, w)
        _, flows_backward = self.compute_flow(aligned_stack_for_flow)
        flows_backward = flows_backward.view(-1, 2, h_f, w_f)

        # 3.2 准备参考特征 (第 4 帧, index 3)
        ref_feat = feats_[:, 3:4, :, :, :].repeat(1, t, 1, 1, 1).view(-1, *feats_.shape[-3:])
        oth_feat = feats_.view(-1, *feats_.shape[-3:])
        
        # 3.3 粗对齐 (Warp)
        oth_feat_warped = flow_warp(oth_feat, flows_backward.permute(0, 2, 3, 1))
        
        # 3.4 精对齐 (DCN)
        # 所有 7 帧（包括参考帧自己）都会通过 DCN
        # 对于参考帧，光流为 0，DCN 将起到自适应特征增强的作用
        aligned_feat = _checkpoint_wrapper(self.dcn_alignment, oth_feat, ref_feat, oth_feat_warped, flows_backward)
        
        # 恢复形状 (n, 7, mid_channels, h_f, w_f)
        return aligned_feat.view(n, t, -1, h_f, w_f)


class ResidualBlocksWithInputConv(nn.Module):
    """带输入卷积的残差块堆叠
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数，默认64
        num_blocks (int): 残差块数量，默认30
    """
    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = [
            # 输入卷积（通道匹配）
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            # 残差块堆叠
            make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels)
        ]
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)

class ResidualBlockNoBN(nn.Module):
    """无批归一化的残差块
    
    Args:
        mid_channels (int): 中间通道数，默认64
        res_scale (float): 残差缩放因子，默认1
    """
    def __init__(self, mid_channels=32, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.init_weights()

    def init_weights(self):
        # 使用Kaiming初始化并缩小初始权重
        init_weights(self.conv1, init_type='kaiming')
        init_weights(self.conv2, init_type='kaiming')
        self.conv1.weight.data *= 0.1
        self.conv2.weight.data *= 0.1

    def forward(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return identity + x * self.res_scale  # 残差连接

class DeformableAlignment(nn.Module):
    """可变形对齐模块
    
    Args:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel (int): 卷积核大小，默认3
        padding (int): 填充大小，默认1
        deform_groups (int): 可变形卷积组数，默认8
        max_residue_magnitude (int): 最大偏移幅度，默认10
    """
    def __init__(self, in_channels, out_channels, kernel=3, padding=1, 
                 deform_groups=8, max_residue_magnitude=10, use_checkpoint=False):
        super().__init__()
        self.max_residue_magnitude = max_residue_magnitude
        
        # 偏移量预测网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * out_channels + 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, 27 * deform_groups, 3, 1, 1),  # 输出偏移和掩码
        )
        
        # 可变形卷积
        self.deform_conv = ops.DeformConv2d(
            in_channels, out_channels, kernel_size=kernel,
            stride=1, padding=padding, groups=deform_groups)
        
        self.init_offset()
        self.use_checkpoint=use_checkpoint

    def init_offset(self):
        # 最后卷积层初始化为零
        init_weights(self.conv_offset[-1], init_type='constant')

    def forward(self, cur_feat, ref_feat, warped_feat, flow):
        # 拼接特征和光流作为额外信息
        def _checkpoint_wrapper(module, *args):
            if self.use_checkpoint and self.training:
                # use_reentrant=False 是 PyTorch 推荐的更稳定的写法
                return checkpoint(
                    lambda x, off, m: self.deform_conv(x, off, mask=m), 
                    cur_feat, offset, mask, 
                    use_reentrant=False
                    )
            return module(*args)
        extra_feat = torch.cat([warped_feat, ref_feat, flow], dim=1)
        out = self.conv_offset(extra_feat)
        
        # 分割偏移量和掩码
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        # 限制偏移量范围并叠加原始光流
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset += flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        mask = torch.sigmoid(mask)  # 归一化掩码
        
        # 应用可变形卷积
        return _checkpoint_wrapper(self.deform_conv, cur_feat, offset, mask)
        

class SPyNet(nn.Module):
    """SPyNet光流估计网络
    
    Args:
        pretrained (str): 预训练权重路径，默认None
    """
    def __init__(self):
        super().__init__()
        # 6级基础模块
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        
        # 图像归一化参数
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """核心光流计算流程"""
        n, _, h, w = ref.size()
        # 图像金字塔构建
        ref_pyramid = [ref]
        supp_pyramid = [supp]
        for level in range(5):
            ref_pyramid.append(F.avg_pool2d(ref_pyramid[-1], 2, 2, count_include_pad=False))
            supp_pyramid.append(F.avg_pool2d(supp_pyramid[-1], 2, 2, count_include_pad=False))
        
        # 从粗到细处理
        ref_pyramid = ref_pyramid[::-1]
        supp_pyramid = supp_pyramid[::-1]
        
        flow = ref_pyramid[0].new_zeros(n, 2, h//32, w//32)
        for level in range(len(ref_pyramid)):
            # 上采样光流并优化残差
            if level > 0:
                flow_up = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            else:
                flow_up = flow
            
            # 构建输入：参考帧、变形后的支撑帧、上采样光流
            warped_supp = flow_warp(supp_pyramid[level], flow_up.permute(0, 2, 3, 1), padding_mode='border')
            input_tensor = torch.cat([ref_pyramid[level], warped_supp, flow_up], 1)
            flow = flow_up + self.basic_module[level](input_tensor)
        
        return flow

    def forward(self, ref, supp):
        """前向传播，处理任意尺寸输入"""
        # 调整尺寸到32的倍数
        h, w = ref.shape[2:]
        w_up = w if w % 32 == 0 else 32 * (w//32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h//32 + 1)
        
        ref = F.interpolate(ref, (h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(supp, (h_up, w_up), mode='bilinear', align_corners=False)
        
        # 计算光流并调整回原始尺寸
        flow = F.interpolate(
            self.compute_flow(ref, supp), 
            size=(h, w), mode='bilinear', align_corners=False)
        
        # 调整光流值到原始尺寸比例
        flow[:, 0, :, :] *= w / w_up
        flow[:, 1, :, :] *= h / h_up
        return flow

class SPyNetBasicModule(nn.Module):
    """SPyNet基础模块"""
    def __init__(self):
        super().__init__()
        self.basic_module = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3)
        )

    def forward(self, tensor_input):
        return self.basic_module(tensor_input)

# 模拟配置参数
class Opt:
    isTrain = True  # 训练模式标志A


class flow_restormer(nn.Module):
    def __init__(self, in_chans=64,
                 embed_dim=60, dim = 48,num_blocks = [1,1,1,1],num_refinement_blocks = 1,heads = [1,2,2,2],
                 ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'withBias',
                 use_checkpoint=False
                 ):
        super(flow_restormer, self).__init__()

        num_in_ch = in_chans
        self.num_out_ch = 48
        self.embed_dim=embed_dim
        self.dim=dim
        self.num_blocks=num_blocks
        self.num_refinement_blocks=num_refinement_blocks
        self.heads=heads
        self.ffn_expansion_factor=ffn_expansion_factor
        ################################### 1. Feature Extraction Network ###################################
        # coarse feature
        opt=Opt()
        self.align=TMRNet(opt, use_checkpoint=use_checkpoint)

        #self.skipup = nn.Conv2d(num_in_ch, dim*4, 3, 1, 1, bias=True)
        #self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_f1=nn.Conv2d(num_in_ch, dim, kernel_size=3, stride=1, padding=1)
        # spatial attention module
        self.conv_first = nn.Conv2d(num_in_ch*7, dim, kernel_size=3, stride=1, padding=1)#num_in_ch*9, kernel_size=4, stride=2, nn.ConvTranspose2d
        ################################### 2. HDR Reconstruction Network ###################################
        self.reconstruct=Restormer(
            inp_channels=embed_dim, 
            out_channels=self.num_out_ch, 
            dim = self.dim,
            num_blocks = self.num_blocks, 
            num_refinement_blocks = self.num_refinement_blocks,
            heads = self.heads,
            ffn_expansion_factor =self.ffn_expansion_factor,
            bias = bias,
            LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
            dual_pixel_task = False,       ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
            use_checkpoint=use_checkpoint
        )
        self.use_checkpoint=use_checkpoint

        self.conv_last = nn.Conv2d(self.num_out_ch, 3, 3, 1, 1)

    def forward(self, x):
        #输入是 b t c h w,输出是 n t c h w
        _, t, _, _, _ = x.size()
        if t < 7:
            first_frame = x[:, 0:1, ...].repeat(1, (7-t)//2, 1, 1, 1)
            last_frame = x[:, -1:, ...].repeat(1, (8-t)//2, 1, 1, 1)
            x = torch.cat([first_frame, x, last_frame], dim=1).contiguous()
        elif t > 7:
            start_idx = (t - 7) // 2
            x = x[:, start_idx : start_idx + 7, ...].contiguous()
        # align_feats=self.align(x)
        if self.use_checkpoint and self.training:
            # 包装 reconstruction 过程
            align_feats = checkpoint(self.align, x, use_reentrant=False)
            # align_feats=self.align(x)
        else:
            align_feats=self.align(x)
        # 沿时间维度分割
        split_tensors = torch.chunk(align_feats, 7, dim=1)  # 返回包含9个形状为 (n, 1, c, h, w) 的张量列表
        # 去除时间维度（压缩第2维）
        f1, f2, f3, f4, f5, f6, f7 = [t.squeeze(dim=1) for t in split_tensors]

        x = self.conv_first(torch.cat((f1,f2,f3,f4,f5,f6,f7), dim=1))
        f1=self.conv_f1(f4)
        # CTBs for HDR reconstruction
        #res = self.conv_after_body(self.forward_features(x) + x)
        # res = self.reconstruct(x)
        if self.use_checkpoint and self.training:
            res = checkpoint(self.reconstruct, x, use_reentrant=False)
        else:
            res = self.reconstruct(x)
        x = self.conv_last(f1 + res)
        x = torch.sigmoid(x)
        return x

