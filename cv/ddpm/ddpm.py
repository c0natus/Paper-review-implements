import torch
import torch.nn as nn


class Diffusion():
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device='cuda'):
        super(Diffusion, self).__init__()
        """
        Image에 noise를 추가하는 class.
        noise_steps : noise를 추가하는 횟수, DDPM 논문에 적힌 1000 사용.
        beta_start  : Beta 시작 값, DDPM 논문에 적힌 1e-4 사용.
        beta_end    : Beta 마지막 값, DDPM 논문에 적힌 0.02 사용.
        img_size    : 고정된 img_size, Resource에 맞게 적당한 값 사용.
        devide      : CPU or GPU
        """

        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.noise_schedule().to(device)
        self.alpha = 1.0 - self.beta # DDPM 논문 수식 4 윗줄 참고.
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) # DDPM 논문 수식 4 윗줄 참고.

    def noise_schedule(self):
        """
        각 step에서의 beta 값 계산.
        DDPM 논문대로 linear하게 설정.
        """
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def noise_images(self, x_0, t):
        """
        원본 image로 t step 거친 noise image 얻는 함수.
        DDPM 논문 수식 4 참고.
        x_0 : 원본 image    (batch_size, 3, height, width)
        t   : noise step    (batch_size, )
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        # sqrt_alpha_hat.shape = (batch_size, 1, 1, 1)
        # Batch size에서 각 image마다 서로 다른 timestep을 적용시킨다.
        # 즉, image1에 step1, image2에 step2, ...
        # 그리고 한 image내의 channel, width, height pixel 값에 
        # 같은 timestep을 적용하기 위해 차원을 맞춘다.

        mean_ = sqrt_alpha_hat * x_0
        std_ = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x_0) 
        # (batch, 3, height, width) 만큼의 값을 표준정규분포에서 sampling.
        # 각 pixel마다 sampling을 한다.
        
        noised_images = mean_ + std_ * noise # reparameterization

        return noised_images, noise
    
    def sample_timesteps(self, batch_size: int) -> torch.tensor:
        """
        Batch size만큼 timestep을 sampling한다.
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size,))
    
    def sample_new_image(self, model, n):
        """
        총 n개의 image를 pure gaussion distribution으로부터 sampling한다.
        그리고 학습된 모델을 통해 img를 생성한다.
        DDPM 논문 Algorithm 2 참고
        model   : U-net
        n       : 생성하고 싶은 img 개수
        """

        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            # n개의 image를 smapling 하기 때문에 n개의 
            # (3, height, width)의 표준 정규분포를 sampling한다.

            for idx in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * idx).long().to(self.device)
                predicted_noise = model(x, t) # e_\theta
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if idx > 1: noise = torch.randn_like(x)
                else: noise = torch.zeros_like(x) # reconstruction: variance을 없앤다.
                
                x = 1 / torch.sqrt(alpha) * \
                    (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * predicted_noise) \
                    + torch.sqrt(beta) * noise
                
        model.train()        
        x = (x.clamp(-1, 1) + 1) / 2 
        # tensor의 값이 0 ~ 1 사이의 값을 가지도록 한다.
        # x.clamp           : -1 ~ 1
        # x.clamp + 1       : 0 ~ 2
        # (x.clamp + 1 / 2) : 0 ~ 1
        x = (x * 255).type(torch.uint8)
        # 0 ~ 255 즉, RGB 값을 가지도록 한다.
        return x


class PosEmb(nn.Module):
    def __init__(self, time_ch):
        super(PosEmb, self).__init__()
        """
        Sinusoidal Encoding
        Args
            time_ch    : embedding dimension, ex) 256
        """
        self.time_ch = time_ch
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_ch // 4, self.time_ch),
            nn.SiLU(), # Swish activation.
            nn.Linear(self.time_ch, self.time_ch),
        )

    def forward(self, t):
        half_dim = self.time_ch // 8
        w_k = 1.0 / (
            10000
            ** (torch.arange(0, half_dim, 1, device=t.device).float() / (half_dim-1))
        )

        half_emb = t.repeat(1, half_dim)
        pos_sin = torch.sin(half_emb * w_k)
        pos_cos = torch.cos(half_emb * w_k)
        pos_enc = torch.cat([pos_sin, pos_cos], dim=-1)

        emb = self.time_mlp(pos_enc)
        return emb
    
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch,
                n_groups=32, dropout=0.1):
        super(ResBlock, self).__init__()
        """
        Residual Convolution: DDPM에 따라 2개 Resnet block 사용
        Args
            in_ch   : Input channels의 수
            out_ch  : Output channels의 수
            time_ch : Time step (t) embedding dimension.
            n_groups: the number of groups for group normalization
            dropout : the dropout rate
        """

        self.first_conv = nn.Sequential(
            nn.GroupNorm(n_groups, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), # 크기 유지, channel 늘림
        )

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_ch, out_ch),
        )

        self.second_conv = nn.Sequential(
            nn.GroupNorm(n_groups, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), # 크기 유지, channel 유지.
        )


        if in_ch != out_ch: self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else: self.res_conv = nn.Identity()

    def forward(self, x, time_emb):
        """
        x           : (Batch, in_ch, height, weight)
        time_emb    : (Batch, time_ch)
        """
        h = self.first_conv(x)
        h = h + self.time_mlp(time_emb)[:,:,None, None] # 각 Batch, 각 channel 마다 다른 pos emb 더해줌.
        h = self.second_conv(h)
        h = h + self.res_conv(x) # Residual connection
        return h


class AttentionBlock(nn.Module):
    def __init__(self, in_ch, n_heads=1, n_groups=32, d_k=None):
        super(AttentionBlock, self).__init__()
        """
        Attention Block
        Args
            in_ch       : the number of channels in the input
            n_heads     : the number of heads in multi-head attention
            d_k         : the number of dimensions in each head
            n_groups    : the number of groups for group normalization
        """
        self.n_heads = n_heads

        if d_k is None: d_k = in_ch // n_heads # in_ch // n_heads가 되어야 하지 않나??
        self.d_k = d_k
        self.scale = d_k ** -0.5 # scale for dot-product
        self.norm = nn.GroupNorm(n_groups, in_ch)
        self.projection = nn.Linear(in_ch, n_heads * d_k * 3) # 3 for query, key, value
        self.output = nn.Linear(n_heads * d_k, in_ch)
    
    def forward(self, x, t=None):
        # t는 쓰이지만, resnet과의 입력을 같도록 만들기 위해 넣어줌.
        # U Net의 forward에서 down_layers를 참고.
        
        batch_size, in_ch, height, width = x.shape
        
        x = x.reshape(batch_size, in_ch, -1).permute(0, 2, 1)
        # (batch, in_ch, h||w) -> (batch, h||w, in_ch)

        qkv = self.projection(x).reshape(batch_size, -1, self.n_heads, 3 * self.d_k)
        # self.projection   : 1x1 conv 느낌으로 하나의 pixel에 대응하는 in_ch를 input으로 받음.
        #                   : (batch, h||w, in_ch) -> (batch, h||w, n_heads * d_k * 3)
        # ~.reshape(~)         : (batch, h||w, heads, 3 * d_k)

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # 각 pixel에 대해 head 수만큼 query, key, value 값을 얻는다.
        # q, k, v: (batch, h||w, n_heads, d_k)

        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # einsum("dimension notation of A, dimension notation of B,...->Result Dimension", A, B, ...)
        # (batch, h||w, h||w, heads)
        # bihd: (batch, query_h||w, heads, d_k)
        # bjhd: (batch, key_h||w, heads, d_k)
        # bijh: (batch, query_h||w, attn_h||w, heads)
        #   -> 각 pixel이 query로 쓰일 때 서로 다른 attention 값을 가지게 된다.
        attn = attn.softmax(dim=2)
        # scaled dot-product    : attention 값을 얻는다.
        # dim=-1로 줬을 때 학습이 안 됐음....


        out = torch.einsum('bijh, bjhd->bihd', attn, v)
        # attn의 값에 따라 value의 embedding을 더한다.
        # bijd: (batch, query_h||w, attn, heads)
        # bjhd: (batch, value_h||w, heads, d_k)
        # bihd: (batch, agg_h||w, heads, d_k )
        out = out.reshape(batch_size, -1, self.n_heads * self.d_k)
        # (batch, agg_h||w, heads, d_k ) -> (batch, agg_h||w, heads * d_k)
        out = self.output(out)
        # 각 head의 정보를 합친다.
        # (batch, agg_h||w, heads * d_k) -> (batch, agg_h||w, in_ch)
        out = out + x # residual connetion
        out = out.permute(0, 2, 1).reshape(batch_size, in_ch, height, width)
        return out       


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, has_attn):
        super(DownBlock, self).__init__()
        """
        U-Net에서 왼쪽 부분. 즉, resolution을 낮추는 부분.
        PixelSNAIL의 구조를 참고해야 할 것 같다.
        Args
            in_ch       : input channels 수
            out_ch      : output channels 수
            time_ch     : time embedding dimension
            has_attn    : self-attention module이 필요한 지를 알려줌.
        """
        self.res = ResBlock(in_ch, out_ch, time_ch)
        if has_attn: self.attn = AttentionBlock(out_ch)
        else: self.attn = nn.Identity()
    
    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class MiddleBlock(nn.Module):
    def __init__(self, n_ch, time_ch):
        super(MiddleBlock, self).__init__()
        """
        U-Net에서 가운데 부분. 즉, 제일 아랫 부분.
        Channel 수는 유지한다.
        Args
            n_ch       : channels 수
            time_ch     : time embedding
            has_attn    : self-attention module이 필요한 지를 알려줌.
        """
        self.res1 = ResBlock(n_ch, n_ch, time_ch)
        self.attn = AttentionBlock(n_ch)
        self.res2 = ResBlock(n_ch, n_ch, time_ch)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x
    

class UpBlock(nn.Module):
    def __init__(self, prev_ch, cat_ch, out_ch, time_ch, has_attn):
        super(UpBlock, self).__init__()
        """
        U-Net에서 오른쪽 부분. 즉, resolution을 올리는 부분.
        Args
            prev_ch     : Upblock에서 이전 feature의 channels 수
            cat_ch      : Concate 되는 feature의 channels 수
            out_ch      : output channels 수
            time_ch     : time embedding
            has_attn    : self-attention module이 필요한 지를 알려줌.
        """
        self.res = ResBlock(prev_ch + cat_ch, out_ch, time_ch)
        # U-Net에서 왼쪽의 결과를 concate해서 사용해야 한다.
        # 따라서 input channel을 in_ch + out_ch로 둔다.

        if has_attn: self.attn = AttentionBlock(out_ch)
        else: self.attn = nn.Identity()
    
    def forward(self, x, t):
        x = self.res(x, t)
        x = self.attn(x)

        return x


class Upsample(nn.Module):
    def __init__(self, n_ch):
        super(Upsample, self).__init__()
        """
        U-Net의 오른쪽 부분에서 resolution을 높일 때 사용.
        2배로 높임.
        n_ch    : channel의 수
        """
        self.conv = nn.ConvTranspose2d(
            in_channels=n_ch, 
            out_channels=n_ch, 
            kernel_size=4,
            stride=2,
            padding=1,
        ) # (h, w) -> (4 + (h-1) * 2 - 2 = 2h, 2w)
        # (h, w)
        #   -> ((ker + (h - 1) * strd - 2 * pad), (ker + (w - 1) * strd - 2 * pad))
        # because (x - 1) * s + k = 2 * p + w^\prime
        # hence w^\prime = (x - 1) * s + k - 2 * p

    def forward(self, x, t=None):
        return self.conv(x) 


class Downsample(nn.Module):
    def __init__(self, n_ch):
        super(Downsample, self).__init__()
        """
        U-Net의 왼쪽 부분에서 resolution을 낮출 때 사용.
        2배로 낮춤.
        n_ch    : channel의 수
        """
        self.conv = nn.Conv2d(n_ch, n_ch,
            kernel_size=3,
            stride=2,
            padding=1,
            ) # (h, w) -> ((h-1)//2 + 1 = h//2, w//2) 
        # (h, w) 
        #   -> ((h - ker + 2 * pad) // strd + 1, (w - ker + 2 * pad) // strd + 1)

    def forward(self, x, t=None):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, img_ch=3, init_ch=64, n_groups=8,
            ch_mults=(1, 2, 4, 8), is_attn=(False, False, True, True), n_blocks=2):
        super(UNet, self).__init__()
        """
        Noise 예측 함수 (e_\theta)
        Args
            img_ch    : img channel 수
            init_ch   : 초기 feature map의 channel 수
            ch_mults        : 각 resolution에서의 channel // ch_mults[i] * init_ch
                              ex) 64 -> 128 -> 256 -> 512
            n_blocks        : 각 resolution에서의 block (2 conv + 1 attn)의 수
        """
        n_resolutions = len(ch_mults) 
        # UNet에서 down sample하는 횟수 즉, resolution의 수


        self.img_proj = nn.Conv2d(img_ch, init_ch, kernel_size=7, padding=3)

        time_ch = init_ch * ch_mults[-1]
        self.pos_embed = PosEmb(time_ch) # PosEmb(time_ch)

        # Downsampling
        out_ch = in_ch = init_ch
        self.down_layers = nn.ModuleList([])
        for idx in range(n_resolutions):
            out_ch = init_ch * ch_mults[idx]

            for _ in range(n_blocks):
                self.down_layers.append(
                    DownBlock(in_ch, out_ch, time_ch, is_attn[idx])
                )
                in_ch = out_ch
            if idx < n_resolutions - 1:
                self.down_layers.append(Downsample(out_ch))
        
        # Middle
        self.middel = MiddleBlock(out_ch, out_ch)

        # Upsampling
        # UNet에서 concate하는 방법이 다양하게 있는 것 같은데, 일단 논문의 공식 깃헙을 따랐다.
        # 이외에도 star를 많이 받은 pytorch 깃헙 등 다른 방법을 사용해도 무방한 것 같다.
        self.up_layers = nn.ModuleList([])
        prev_ch = cat_ch = out_ch
        for idx in reversed(range(n_resolutions)):
            out_ch = init_ch * ch_mults[idx]
            for _ in range(n_blocks):
                self.up_layers.append(UpBlock(prev_ch, cat_ch, out_ch, time_ch, is_attn[idx]))
                prev_ch = out_ch
            if idx > 0:
                cat_ch = init_ch * ch_mults[idx-1]
                self.up_layers.append(UpBlock(prev_ch, cat_ch, out_ch, time_ch, is_attn[idx]))
                self.up_layers.append(Upsample(out_ch))
            else:
                self.up_layers.append(UpBlock(prev_ch, cat_ch, out_ch, time_ch, is_attn[idx]))

        # End
        self.end = nn.Sequential(
            nn.GroupNorm(n_groups, init_ch),
            nn.SiLU(),
            nn.Conv2d(init_ch, img_ch,
                kernel_size=3,
                padding=1,)
        )

    def forward(self, x, t):
        """
        x   : (batch_size, 3, img_size, img_size)
        t   : (batch_size, )
        """
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_embed(t)
        x = self.img_proj(x)

        h = [x] # down sampling의 결과를 up sampling에 concate하기 위해 저장.

        for m in self.down_layers:
            x = m(x, t)
            h.append(x)

        x = self.middel(x, t)

        hs_idx = len(h) - 1
        for m in self.up_layers:
            if isinstance(m, Upsample): x = m(x, t)
            else:
                s = h[hs_idx]
                hs_idx -= 1
                x = torch.cat((x, s), dim=1)
                x = m(x, t)
        
        del h

        return self.end(x)