import torch
from torch import nn
from PromptUNet.ImageEncoder import *
from PromptUNet.PromptEncoder import *
import math

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(out_c,affine=False)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_c, affine=False)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)

        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class PromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[2, 4, 8,16,],attention_kernels=[3,4,4,4], d_model=64,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,box=False,use_mlp=False ):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2,use_mlp = use_mlp)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,)
        self.e2 = encoder_block(base_c, base_c * kernels[0], )
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], )
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], )




        """ Bottleneck """
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],)




        self.promptImageCrossAttention_b = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in,num_heads, attention_kernels[1], dropout, )
        self.promptImageCrossAttention_d1 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in//2, num_heads, attention_kernels[2],dropout,)
        self.promptImageCrossAttention_d2= ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4, num_heads, attention_kernels[3], dropout,)

        """ Decoder """
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1], )
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0], )
        self.d4 = decoder_block(base_c * kernels[0], base_c, )

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')

        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """

        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3,p3 = self.e3(p2)
        s4,p4 = self.e4(p3)



        b = self.b(p4)
        """never managed to determine whether it is better to input altered prompts into subsequent ImPFusion blocks or just the original prompt outputs from the PromptEncoder each time, change 'prompts_' to 'prompts' to pass forward altered prompts"""
        if train_attention:
            b_prompted,prompts_ = self.promptImageCrossAttention_b(b,prompts,original_prompts)
            b = b + b_prompted

        d1 = self.d1(b, s4)

        if train_attention:
            d1_prompted, prompts_ = self.promptImageCrossAttention_d1(d1, prompts,original_prompts)
            d1 = d1 + d1_prompted


        d2 = self.d2(d1, s3)

        if train_attention:
            d2_prompted,prompts_ = self.promptImageCrossAttention_d2(d2,prompts,original_prompts)
            d2 = d2 + d2_prompted

        d3 = self.d3(d2, s2)


        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs


class MaskedPromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[2, 4, 8,16,],attention_kernels=[3,4,4,4], d_model=64,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1, box=False,use_mlp=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,)
        self.e2 = encoder_block(base_c, base_c * kernels[0], )
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], )
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], )
        #self.e5 = encoder_block(base_c*kernels[2],base_c*kernels[3],norm_name)
        """ Bottleneck """
        #self.b1 = conv_block(base_c * kernels[3], base_c * kernels[4], norm_name)
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],)


        self.promptImageCrossAttention_b = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in,num_heads,attention_kernels[1],dropout,)
        self.promptImageCrossAttention_d1 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in//2, num_heads, attention_kernels[2],dropout,)
        self.promptImageCrossAttention_d2= maskedImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads, attention_kernels[3], dropout,radius=48)
        #self.promptIMageCrossAttention_final = maskedImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,radius=40)
        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3], norm_name)
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1], )
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0], )
        self.d4 = decoder_block(base_c * kernels[0], base_c, )

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):

        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """

        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)


        b = self.b(p4)


        if train_attention:
            b_prompted, promptsd1 = self.promptImageCrossAttention_b(b, prompts,original_prompts,)
            b = b+ b_prompted



        d1 = self.d1(b, s4)

        if train_attention:
            d1_prompted, prompts_ = self.promptImageCrossAttention_d1(d1, promptsd1,original_prompts,)
            d1 = d1 + d1_prompted



        d2 = self.d2(d1, s3)

        if train_attention:
            d2_prompted,prompts = self.promptImageCrossAttention_d2(d2,prompts,original_prompts,points)
            d2 =  d2 +d2_prompted

        d3 = self.d3(d2, s2)


        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs
class pointLoss(nn.Module): #custom loss function to ensure dependence on user points, essentially gaussian weighted CE + mask over non-relevent classes

    def __init__(self,radius):
        super().__init__()
        self.radius = radius
        self.sigma = radius//3
        self.init_loss = nn.CrossEntropyLoss(reduction='none')


    def gaussian(self,i, j, i_p, j_p,):

        # i1, j1: the coordinates of the pixel
        # i, j: the coordinates of the center
        # sigma: the standard deviation of the Gaussian
        # returns the value of the Gaussian function at pixel i1,j1
        d = torch.sqrt((i - i_p) ** 2 + (j - j_p) ** 2)  # distance between pixel and center
        f = torch.exp(-0.5 * (d / self.sigma) ** 2)   # Gaussian function reweighted s.t. it is 1 at d=0
        return f

    def forward(self,y_pred,y_true,points,point_labels,device):

        B,L,_ = points.shape
        _,_,H,W = y_true.shape
        y_pred = y_pred.softmax(dim=1)
        mask = torch.zeros((B,H,W)).to(device)
        y_true_one_hot = torch.nn.functional.one_hot(y_true.long(),num_classes=3).squeeze()
        #shape is BxHxWxC now
        y_true_one_hot = torch.permute(y_true_one_hot,(0,3,1,2))
        for b in range(B):
            for l in range(L):
                i_p,j_p = points[b,l,:]
                i_min = max(0,i_p-self.radius//2)
                i_max = min(H,i_p + 1 + self.radius//2)

                j_min = max(0,j_p-self.radius//2)
                j_max = min(W,j_p + self.radius//2)

                i_grid = list(range(int(i_min),int(i_max)))
                j_grid = list(range(int(j_min),int(j_max)))

                for i in i_grid:
                    for j in j_grid:

                        if y_true[b,0,i,j] == point_labels[b,l,0]:
                            mask[b,i,j] += self.gaussian(i,j,i_p,j_p)

        loss = self.init_loss(y_true_one_hot.to(torch.float32),y_pred)


        loss*=mask

        return torch.mean(loss)

class SymmetricPromptUNet(nn.Module): #unet w/ cross attention in encoder also, performs poorly
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,],attention_kernels=[3,2,4,5], d_model=384,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,batch_norm=False,box=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,)
        self.e2 = encoder_block(base_c, base_c * kernels[0],)

        self.e3 = conv_block(base_c * kernels[0], base_c * kernels[1],)
        self.e3_p = torch.nn.MaxPool2d((2,2))
        self.e4 = conv_block(base_c * kernels[1], base_c * kernels[2],)
        self.e4_p = torch.nn.MaxPool2d((2, 2))


        """ Bottleneck """

        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],)

        self.promptImageCrossAttention_e3 = ImageEncoder(self.device, self.pe_layer, d_model=self.d_model,d_image_in=self.d_image_in // 4, num_heads=num_heads,num_blocks=5, dropout=dropout,)
        self.promptImageCrossAttention_e4 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 2,num_heads, 3, dropout, )

        self.promptImageCrossAttention_d1 = ImageEncoder(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = 3,dropout = dropout)
        self.promptImageCrossAttention_d2 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads,5, dropout,)
        #self.crossattention_final = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,)
        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3])
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1],)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0])
        self.d4 = decoder_block(base_c * kernels[0], base_c,)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')
        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """
        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)

        s3 = self.e3(p2)
        if train_attention:
            s3_prompted,prompts_d2 = self.promptImageCrossAttention_e3(s3,prompts,original_prompts)
            s3 = s3 + s3_prompted
        p3 = self.e3_p(s3)

        s4 = self.e4(p3)
        if train_attention:
            s4_prompted, prompts_d1 = self.promptImageCrossAttention_e4(s4,prompts,original_prompts)
            s4 = s4 + s4_prompted
        p4 = self.e4_p(s4)
        """ Bottleneck """

        b = self.b(p4)


        """ Decoder """

        d1 = self.d1(b, s4)
        if train_attention:
            d1_prompted, prompts = self.promptImageCrossAttention_d1(d1, prompts_d1,original_prompts)
            d1 = d1 + d1_prompted

        d2 = self.d2(d1, s3)
        if train_attention:
            d2_prompted,prompts = self.promptImageCrossAttention_d2(d2,prompts_d2,original_prompts,)
            d2 = d2 + d2_prompted

        d3 = self.d3(d2, s2)

        #if train_attention:
        #    d3,prompts = self.crossattention_final(d3,prompts,original_prompts,points)

        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs




class PromptUNetAttnMap(nn.Module): # returns attention maps, this actually slows down performance so initialise weights from checkpoint with 'strict=False', don't train using this
    def __init__(self,device, in_c, out_c, base_c, kernels=[6, 12, 24,48,],attention_kernels=[3,4,4,4], d_model=384,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,box=False):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c)
        self.e2 = encoder_block(base_c, base_c * kernels[0],)

        # self.e3 = conv_block(base_c * kernels[0], base_c * kernels[1],)
        # self.e3_p = torch.nn.MaxPool2d((2,2))
        # self.e4 = conv_block(base_c * kernels[1], base_c * kernels[2],)
        # self.e4_p = torch.nn.MaxPool2d((2, 2))

        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], )
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], )
        """ Bottleneck """

        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],)

        # self.promptImageCrossAttention_e3 = ImageEncoder(self.device, self.pe_layer, d_model=self.d_model,d_image_in=self.d_image_in // 4, num_heads=num_heads,num_blocks=attention_kernels[1], dropout=dropout,batch=batch_norm)
        # self.promptImageCrossAttention_e4 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 2,num_heads, attention_kernels[2], dropout, batch=batch_norm)
        self.promptImageCrossAttention_b = ImageEncoderMap(self.device, self.pe_layer, self.d_model, self.d_image_in,
                                                         num_heads, attention_kernels[1], dropout, )

        self.promptImageCrossAttention_d1 = ImageEncoderMap(self.device, self.pe_layer, d_model = self.d_model, d_image_in = self.d_image_in//2,num_heads = num_heads, num_blocks = attention_kernels[2],dropout = dropout,)
        self.promptImageCrossAttention_d2 = ImageEncoderMap(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,num_heads,attention_kernels[3], dropout,)
        #self.crossattention_final = ImageEncoder(self.device,self.pe_layer,self.d_model,self.d_image_in//8,num_heads,2,dropout,)
        """ Decoder """
       # self.dnew = decoder_block(base_c * kernels[4], base_c * kernels[3])
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],)
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1],)
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0],)
        self.d4 = decoder_block(base_c * kernels[0], base_c,)

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,attn_maps=False,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')
        #print(points)
        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """
        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)

        s3,p3 = self.e3(p2)
        # if train_attention:
        #     s3_prompted,prompts_d2 = self.promptImageCrossAttention_e3(s3,prompts,original_prompts)
        #     s3 = s3 + s3_prompted
       # p3 = self.e3_p(s3)

        s4,p4 = self.e4(p3)
        # if train_attention:
        #     s4_prompted, prompts_d1 = self.promptImageCrossAttention_e4(s4,prompts,original_prompts)
        #     s4 = s4 + s4_prompted
       # p4 = self.e4_p(s4)
        """ Bottleneck """

        b = self.b(p4)
        if train_attention:
            b_prompted, prompts_ = self.promptImageCrossAttention_b(b,prompts,original_prompts,'b',attn_maps)
            b = b + b_prompted


        """ Decoder """

        d1 = self.d1(b, s4)
        if train_attention:
            d1_prompted, prompts_ = self.promptImageCrossAttention_d1(d1, prompts,original_prompts,'d1',attn_maps)
            d1 = d1+d1_prompted

        d2 = self.d2(d1, s3)
        if train_attention:
            d2_prompted,prompts_ = self.promptImageCrossAttention_d2(d2,prompts,original_prompts,'d2',attn_maps)
            d2 = d2 + d2_prompted

        d3 = self.d3(d2, s2)

        #if train_attention:
        #    d3,prompts = self.crossattention_final(d3,prompts,original_prompts,points)

        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs

    # this is actually just focal loss now
class NormalisedFocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):

        target = target.to(dtype=torch.int64)
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target_one_hot = target.view(-1, 1)

        logpt = torch.nn.functional.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target_one_hot)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        """following code was removed, uncomment it to turn this func into 'normalised focal loss' """
        # normalising_constant = torch.sum((1-pt)**self.gamma)
        #
        # loss /= normalising_constant

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class combine_loss(nn.Module):

    def __init__(self, loss1, loss2, alpha):
        super().__init__()
        self.l1 = loss1
        self.l2 = loss2
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        return self.alpha * (self.l1(y_pred, y_true)) + (1 - self.alpha) * (self.l2(y_pred, y_true))


class combine_point_loss(nn.Module):

    def __init__(self,pointloss,loss,alpha,beta):
        super().__init__()

        self.pointloss = pointloss
        self.loss = loss
        self.alpha = alpha
        self.beta = beta

    def forward(self,y_pred,y_true,points,labels,device):

        loss1 = self.pointloss(y_pred,y_true,points,labels,device)
        loss2 = self.loss(y_pred,y_true)
        pl = self.alpha*loss1
        gl = self.beta*loss2
        print(f'point loss: {pl} gen loss: {gl}')


        return pl+gl




"""not finished yet - going to try placing cross attention in-between conv modules so it goes like res(conv)->res(cross)->res(conv)..."""
class IterativePromptUNet(nn.Module):
    def __init__(self,device, in_c, out_c, base_c, kernels=[2, 4, 8,16,],attention_kernels=[3,4,4,4], d_model=64,num_prompt_heads=4,num_heads=8,img_size=(512,512),dropout=0.1,box=False,use_mlp=False ):
        super().__init__()

        self.pe_layer = PositionEmbeddingRandom(d_model // 2,use_mlp = use_mlp)

        self.d_image_in = base_c*kernels[-1]
        self.d_model = d_model
        self.device = device
        self.promptSelfAttention = PromptEncoder(self.device,self.pe_layer,self.d_model,img_size,num_prompt_heads,attention_kernels[0],dropout,box)
        """ Encoder """
        self.e1 = encoder_block(in_c, base_c,)
        self.e2 = encoder_block(base_c, base_c * kernels[0], )
        self.e3 = encoder_block(base_c * kernels[0], base_c * kernels[1], )
        self.e4 = encoder_block(base_c * kernels[1], base_c * kernels[2], )




        """ Bottleneck """
        self.b = conv_block(base_c * kernels[2], base_c * kernels[3],)
        self.b_it = conv_block(base_c * kernels[3], base_c * kernels[3], )



        self.promptImageCrossAttention_b = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in,num_heads, attention_kernels[1], dropout, )
        self.promptImageCrossAttention_d1 = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in//2, num_heads, attention_kernels[2],dropout,)
        self.promptImageCrossAttention_d2= ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4, num_heads, attention_kernels[3], dropout,)
        self.promptImageCrossAttention_b_it = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in,
                                                        num_heads, attention_kernels[1], dropout, )
        self.promptImageCrossAttention_d1_it = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 2,
                                                         num_heads, attention_kernels[2], dropout, )
        self.promptImageCrossAttention_d2_it = ImageEncoder(self.device, self.pe_layer, self.d_model, self.d_image_in // 4,
                                                         num_heads, attention_kernels[3], dropout, )

        """ Decoder """
        self.d1 = decoder_block(base_c * kernels[3], base_c * kernels[2],)
        self.d1_it = conv_block(base_c*kernels[2],base_c*kernels[2])
        self.d2 = decoder_block(base_c * kernels[2], base_c * kernels[1], )
        self.d2_it = conv_block(base_c * kernels[1], base_c * kernels[1])
        self.d3 = decoder_block(base_c * kernels[1], base_c * kernels[0], )
        self.d4 = decoder_block(base_c * kernels[0], base_c, )

        """ Classifier """
        self.outputs = nn.Conv2d(base_c, out_c, kernel_size=1, padding=0)




    def forward(self, images,points,labels,train_attention = True):
        #print(f'inputs\nimages:{images.device}\npoints{points.device}\nlabels:{labels.device}')

        if train_attention:
            prompts, original_prompts = self.promptSelfAttention(points, labels)

        """ Encoder """

        s1, p1 = self.e1(images)
        s2, p2 = self.e2(p1)
        s3,p3 = self.e3(p2)
        s4,p4 = self.e4(p3)



        b = self.b(p4)

        if train_attention:
            b_prompted,prompts = self.promptImageCrossAttention_b(b,prompts,original_prompts)
            b = b + b_prompted

        d1 = self.d1(b, s4)

        if train_attention:
            d1_prompted, prompts = self.promptImageCrossAttention_d1(d1, prompts,original_prompts)
            d1 = d1 + d1_prompted
            d1 = self.d1_it(d1)
            d1_prompt_it = self.promptImageCrossAttention_d1_it(d1)


        d2 = self.d2(d1, s3)

        if train_attention:
            d2_prompted,prompts = self.promptImageCrossAttention_d2(d2,prompts,original_prompts)
            d2 = d2 + d2_prompted

        d3 = self.d3(d2, s2)


        d4 = self.d4(d3,s1)
        outputs = self.outputs(d4)

        return outputs









