import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import test_dataloader
# from skimage.metrics import peak_signal_noise_ratio
# from skimage.metrics import structural_similarity
from metrics.cal_psnr_ssim import calculate_psnr
from metrics.cal_psnr_ssim import calculate_ssim
import time
from models.LWDeblur import LWDNet
import lpips
import os
class util_of_lpips():
    def __init__(self, net='vgg', use_gpu=True):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
            是否使用GPU,默认不使用
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net=net)
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):
        '''
        Parameters
        ----------
        img1_path : str
            图像1的路径.
        img2_path : str
            图像2的路径.
        Returns
        -------
        dist01 : torch.Tensor
            学习的感知图像块相似度(Learned Perceptual Image Patch Similarity, LPIPS).

        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))
        with torch.no_grad():
            if self.use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
            dist01 = self.loss_fn.forward(img0, img1)
        return dist01

def eval_lpips():
    cal_lpips = util_of_lpips()
    lpips_adder =Adder()
    out_path = "/mnt/d/myPaper/baseline_wo_ffm/visualization/wo"
    gt_path = "/mnt/d/datasets/GoPro_train_latest/test/target"
    img_list = os.listdir(gt_path)
    iter=0
    lpips_total=0
    for img in img_list:
        iter=iter+1
        gt_img = os.path.join(gt_path,img)
        out_img = os.path.join(out_path,img)
        lpips = cal_lpips.calc_lpips(gt_img,out_img)
        lpips_total=lpips_total+lpips
        # lpips_adder(lpips)
        print("iter: %d, lpips: %.4f "%(iter,lpips))
    print("###########################")
    print("Average LPIPS: %.4f"%(lpips_total/iter))
        
def eval(model, data_dir,weight_dir,result_dir=None,save_image=False):
    state_dict = torch.load(weight_dir)
    model.load_state_dict(state_dict['params'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()


        # Hardware warm-up
        # for iter_idx, data in enumerate(dataloader):
        #     input_img, label_img, _ = data
        #     input_img = input_img.to(device)
        #     tm = time.time()
        #     _ = model(input_img)
        #     _ = time.time() - tm

        #     if iter_idx == 20:
        #         break

        # Main Evaluation
        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            tm = time.time()

            pred = model(input_img)[2]

            elapsed = time.time() - tm
            adder(elapsed)

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            if save_image:
                save_name = os.path.join(result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            # psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            # print('%d iter PSNR: %.2f' % (iter_idx + 1, psnr))
            psnr = calculate_psnr(pred_numpy, label_numpy, crop_border=0)
            ssim = calculate_ssim(pred_numpy, label_numpy, crop_border=0)
            psnr_adder(psnr)
            ssim_adder(ssim)
            
            print("%d iter PSNR: %.2f SSIM: %.4f"%(iter_idx+1,psnr,ssim))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f dB' % (ssim_adder.average()))
        # print("Average time: %f" % adder.average())

if __name__ == "__main__":
    eval_lpips()
    # model = LWDNet(12,32)
    # model.cuda()
    # dataset_dir = ""
    # eval(model,data_dir='/mnt/d/datasets/GoPro_train_latest',weight_dir='weights/net_g_590000.pth')