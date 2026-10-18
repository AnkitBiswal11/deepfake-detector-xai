import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PyDeepFakeDet import models
from PyDeepFakeDet.utils.checkpoint import load_checkpoint


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        size=(224, 224),
        num_cls=2,
        mean=None,
        std=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Resolve nested target layer (e.g. net.0.layer4)
        layer = self.model
        for sub in target_layer.split('.'):
            if sub.isdigit():
                layer = layer[int(sub)]
            else:
                layer = getattr(layer, sub)

        layer.register_forward_hook(self.__forward_hook)
        layer.register_full_backward_hook(self.__backward_hook)

        self.size = size
        self.origin_size = None
        self.num_cls = num_cls
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]

        self.grads = []
        self.fmaps = []

    def forward(self, img_arr: np.ndarray, label=None, show=False, write=True, path=''):
        img_input = self.__img_preprocess(img_arr.copy()).to(self.device)

        output = self.model({"img": img_input})
        logits = output["logits"]

        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        verdict = "Fake / Manipulated" if pred_idx == 1 else "Real"
        print(f"Prediction: {verdict} (Confidence: {probs[pred_idx] * 100:.2f}%)")

        self.model.zero_grad()
        loss = self.__compute_loss(logits, label if label is not None else pred_idx)
        loss.backward()

        grads_val = self.grads[0].cpu().data.numpy().squeeze()
        fmap = self.fmaps[0].cpu().data.numpy().squeeze()
        cam = self.__compute_cam(fmap, grads_val)

        cam_show = cv2.resize(cam, self.origin_size)
        img_show = img_arr.astype(np.float32) / 255.0
        self.__show_cam_on_image(img_show, cam_show, if_show=show, if_write=write, path=path)

        self.fmaps.clear()
        self.grads.clear()

    def __img_preprocess(self, img_in: np.ndarray) -> torch.Tensor:
        self.origin_size = (img_in.shape[1], img_in.shape[0])
        img = cv2.resize(img_in.copy(), self.size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transform(Image.fromarray(img)).unsqueeze(0)

    def __backward_hook(self, module, grad_in, grad_out):
        self.grads.append(grad_out[0].detach())

    def __forward_hook(self, module, input, output):
        self.fmaps.append(output)

    def __compute_loss(self, logit, index):
        idx = torch.tensor([[index]], dtype=torch.long, device=self.device)
        one_hot = torch.zeros((1, self.num_cls), device=self.device).scatter_(1, idx, 1.0)
        one_hot.requires_grad = True
        return torch.sum(one_hot * logit)

    def __compute_cam(self, feature_map, grads):
        if feature_map.ndim < 3:
            return np.ones(self.size, dtype=np.float32)
        alpha = np.mean(grads, axis=(1, 2))
        cam = np.zeros(feature_map.shape[1:], dtype=np.float32)
        for k, ak in enumerate(alpha):
            cam += ak * feature_map[k]
        cam = np.maximum(cam, 0)
        cam_min, cam_max = np.min(cam), np.max(cam)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def __show_cam_on_image(self, img: np.ndarray, mask: np.ndarray, if_show=False, if_write=True, path=""):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        cam = heatmap + img
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        if if_write and path:
            out_dir = os.path.dirname(os.path.abspath(path))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(path, cam)
            print(f"Heatmap saved successfully to: {path}")
        if if_show:
            plt.imshow(cam[:, :, ::-1])
            plt.axis("off")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grad-CAM visualization")
    parser.add_argument("--model", type=str, default="ResNet50", help="Model name")
    parser.add_argument("--pth", type=str, default="", help="Path to checkpoint (optional)")
    parser.add_argument("--layer", type=str, default="net.0.layer4", help="Target layer for CAM")
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    parser.add_argument("--save_path", type=str, default="demo/gradcam_output.jpg", help="Path to save heatmap")
    args = parser.parse_args()

    cfg = {"PRETRAINED": True if not args.pth else False, "NUM_CLASSES": 2, "ESCAPE": ""}
    net = getattr(models, args.model)(cfg)

    if args.pth and os.path.exists(args.pth):
        load_checkpoint(args.pth, net, False)
        print(f"Loaded weights from {args.pth}")
    else:
        print("Using base/pretrained weights.")

    img = cv2.imread(args.img, 1)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {args.img}")

    grad_cam = GradCAM(
        net,
        num_cls=2,
        target_layer=args.layer,
        size=(224, 224),
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    grad_cam.forward(img, show=False, write=True, path=args.save_path)