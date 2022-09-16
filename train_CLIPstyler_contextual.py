from PIL import Image
import numpy as np

import torch
import torch.nn
import torch.optim as optim
from torchvision import transforms, models

import StyleNet
import utils
import clip
import torch.nn.functional as F
from template import imagenet_templates

from PIL import Image 
import PIL 
from torchvision import utils as vutils
import argparse
from torchvision.transforms.functional import adjust_contrast

parser = argparse.ArgumentParser()

parser.add_argument('--content_path', type=str, default="./face.jpg",
                    help='Image resolution')
parser.add_argument('--content_name', type=str, default="face",
                    help='Image resolution')
parser.add_argument('--exp_name', type=str, default="exp1",
                    help='Image resolution')
parser.add_argument('--text', type=str, default="Fire",
                    help='Image resolution')
parser.add_argument('--lambda_tv', type=float, default=2e-3,
                    help='total variation loss parameter')
parser.add_argument('--lambda_patch', type=float, default=9000,
                    help='PatchCLIP loss parameter')
parser.add_argument('--lambda_dir', type=float, default=500,
                    help='directional loss parameter')
parser.add_argument('--lambda_patch_context', type=float, default=7000,
                    help='Context loss parameter')
parser.add_argument('--lambda_global_context', type=float, default=20,
                    help='Context loss parameter')

parser.add_argument('--lambda_c', type=float, default=150,
                    help='content loss parameter')
parser.add_argument('--crop_size', type=int, default=128,
                    help='cropped image size')
parser.add_argument('--num_crops', type=int, default=64,
                    help='number of patches')
parser.add_argument('--img_width', type=int, default=512,
                    help='size of images')
parser.add_argument('--img_height', type=int, default=512,
                    help='size of images')
parser.add_argument('--max_step', type=int, default=200,
                    help='Number of domains')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='Number of domains')
parser.add_argument('--thresh', type=float, default=0.7,
                    help='Number of domains')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assert (args.img_width%8)==0, "width must be multiple of 8"
assert (args.img_height%8)==0, "height must be multiple of 8"

VGG = models.vgg19(pretrained=True).features
VGG.to(device)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)
    
def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image,device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

    
def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    
    return loss_var_l2

#Contextual Loss
# https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da
def contextual_loss(x, y, h=0.5):
    """Computes contextual loss between x and y.

    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).

    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    # cx_loss=100.0
    # print(x.shape)    # 1 * 256
    # print(y.shape)    # 1 * 256
    assert x.size() == y.size()
    N, F = x.size()  # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).
    # print(N,F)
    # y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
    y_mu = y.mean(1).reshape(-1,1)
    #
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)
    # print(x_normalized.shape, y_normalized.shape)     # (1,256) and (1,256)
    #
    # # The equation at the bottom of page 6 in the paper
    # # Vectorized computation of cosine similarity for each pair of x_i and y_j
    # x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, F)
    # y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, F)
    x_normalized = x_normalized.reshape(N, -1)  # (N, F)
    y_normalized = y_normalized.reshape(N, -1)  # (N, F)
    # print(x_normalized.shape, y_normalized.shape)     #  (1,256) and (1,256)
    # cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, F, F)

    x_normalized_resize=x_normalized.resize(N, 1,F)
    y_normalized_resize=y_normalized.resize(N, 1, F)
    xT_normalized_resize = x_normalized_resize.transpose(1,2)
    cosine_sim=torch.matmul(xT_normalized_resize, y_normalized_resize)
    # print(cosine_sim.shape)    #(1,256,256)
    d = 1 - cosine_sim  # (N, F, F)  d[n, i, j] means d_ij for n-th data
    # d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, F, 1)
    d_min, _ = torch.min(d, dim=2, keepdim=True)  # (N, F, 1)

    # # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # # Eq(3)
    w = torch.exp((1 - d_tilde) / h)
    # print(w[0][0])   # values range = [0.94----0.98]
    # print("shapes of d = (%d,%d), d_min = (%d,%d), d_tilde = (%d,%d)" %((d.shape), (d_min.shape),(d_tilde.shape)))
    # print(d.shape, d_min.shape, d_tilde.shape)     #  (32,256,256) , (32,256,1), (32,256,256)
    # # Eq(4)
    # cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)  # (N, H*W, H*W)
    # print(cx_ij[0][0])   #value range #0.0038, 0.0039, 0.0040
    # print("shapes of w = (%d,%d), cx_ij = (%d,%d)" %((w.shape),(cx_ij.shape)))
    # print(w.shape, cx_ij.shape)   # (1,256,256) , (1,256,256)

    # # Eq (1)
    # cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )

    # print(type(cx))

    cx = 10000*torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    # print("shapes of cx = (%d,%d) and cx_loss= (%d,%d)" %((cx.shape), (cx_loss.shape)) )
    # print(cx.shape, cx_loss.shape)       # (32) and ([])
    # print("cx=%.2f and cx_loss=%.2f" %(cx, cx_loss) )
    # print(cx, cx_loss)
    return cx_loss


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]

content_path = args.content_path
content_image = utils.load_image2(content_path, img_height=args.img_height,img_width=args.img_width)
content = args.content_name
exp = args.exp_name

content_image = content_image.to(device)

content_features = utils.get_features(img_normalize(content_image), VGG)

target = content_image.clone().requires_grad_(True).to(device)

style_net = StyleNet.UNet()
style_net.to(device)

style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.4,
                 'conv4_1': 0.8,
                 'conv5_1': 1.6}

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

output_image = content_image

m_cont = torch.mean(content_image,dim=(2,3),keepdim=False).squeeze(0)
m_cont = [m_cont[0].item(),m_cont[1].item(),m_cont[2].item()]

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
    transforms.Resize(224)
])
device='cuda'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = args.text

source = "a Photo"

with torch.no_grad():
    template_text = compose_text_with_templates(prompt, imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    template_source = compose_text_with_templates(source, imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)

    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)

    source_features = clip_model.encode_image(clip_normalize(content_image,device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    
num_crops = args.num_crops
for epoch in range(0, steps+1):
    
    scheduler.step()
    target = style_net(content_image,use_sigmoid=True).to(device)
    target.requires_grad_(True)
    
    target_features = utils.get_features(img_normalize(target), VGG)
    
    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)   #Content Loss

    loss_patch=0 
    img_proc =[]
    for n in range(num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc,dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug,device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
    
    img_direction = (image_features-source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    
    text_direction = (text_features-text_source).repeat(image_features.size(0),1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp<args.thresh] =0
    loss_patch+=loss_temp.mean()       # Patch Loss

    glob_features = clip_model.encode_image(clip_normalize(target,device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))
    
    glob_direction = (glob_features-source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)
    
    loss_glob = (1- torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()    #Global Direction Loss

    
    reg_tv = args.lambda_tv*get_image_prior_losses(target)    #Total Variation Loss

    loss_patch_context = 0
    loss_patch_temp = contextual_loss(img_direction, text_direction)
    loss_patch_temp = torch.abs(loss_patch_temp)
    # loss_patch_temp[loss_patch_temp < args.thresh] = 0
    loss_patch_context += loss_patch_temp.mean()  # #Context Patch Loss

    # text_direction_column = text_direction.mean(dim=0)
    # text_direction_column = torch.reshape(text_direction_column, (1,512))
    # loss_glob_context = contextual_loss(glob_direction, text_direction_column)  #Context Global Loss
    # total_loss = args.lambda_patch*loss_patch + content_weight * content_loss+ reg_tv+ args.lambda_dir*loss_glob +  args.lambda_patch_context*loss_patch_context + args.lambda_global_context*loss_glob_context

    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob + args.lambda_patch_context * loss_patch_context
    total_loss_epoch.append(total_loss)     #Total Loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("After %d criterions:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', loss_patch.item())
        print('dir loss: ', loss_glob.item())
        print('TV loss: ', reg_tv.item())
        print('Patch Context Loss', loss_patch_context.item())
        # print('Global Context Loss', loss_glob_context.item())
    
    if epoch %50 ==0:
        out_path = './outputs/'+prompt+'_'+content+'_'+exp+'.jpg'
        output_image = target.clone()
        output_image = torch.clamp(output_image,0,1)
        output_image = adjust_contrast(output_image,1.5)
        vutils.save_image(
                                    output_image,
                                    out_path,
                                    nrow=1,
                                    normalize=True)

