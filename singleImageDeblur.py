from models.LWDeblur import LWDNet
import torch
import cv2
import torchvision.transforms.functional as tf

def read_img(img_path):
    """
    return:
        c h w
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype("float32")
    img = torch.from_numpy(img.transpose(2,0,1)).float()/255.0
    return img

def save_img(tensor,save_path):
    """
    tensor:
        c h w
    """
    image = tf.to_pil_image(tensor)
    image.save(save_path)

def simgleImageDeblur(blur_path,deblur_path,weight_dir):
    blur_img = read_img(blur_path).unsqueeze(0).to("cuda")

    state_dict = torch.load(weight_dir)
    model = LWDNet(12,32).cuda()
    model.load_state_dict(state_dict['params'])

    model.eval()

    clear_img = model(blur_img)[2][0]
    save_img(clear_img,deblur_path)


if __name__ == "__main__":
    simgleImageDeblur("realblur/patch_6.jpg","realblur/patch_6_deblur.png",'weights/net_g_590000.pth')




