import argparse
from PIL import Image
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils.dataset import Padding
from quantization.quantize import ptq_serving, qat_serving


classes = {
    0: '2%', 1: '박카스', 2: '칠성 사이다', 3: '칠성 사이다 제로', 4: '초코 우유',
    5: '코카 콜라', 6: '데미소다 사과', 7: '데미소다 복숭아', 8: '솔의눈', 9: '환타 오렌지',
    10: '게토레이', 11: '제티', 12: '맥콜', 13: '우유', 14: '밀키스', 15: '밀키스 제로',
    16: '마운틴 듀', 17: '펩시', 18: '펩시 제로', 19: '포카리 스웨트', 20: '파워에이드',
    21: '레드불', 22: '식혜', 23: '스프라이트', 24: '스프라이트 제로', 25: '딸기 우유',
    26: '비타 500', 27: '브이톡 블루레몬', 28: '브이톡 복숭아', 29: '웰치스 포도',
    30: '웰치스 오렌지', 31: '웰치스 화이트그레이프',32: '제로 콜라',
}


transformation = transforms.Compose([
    Padding(fill=(0, 0, 0)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def load_image(src: str):    
    img = Image.open(src).convert('RGB')
    img = transformation(img)
    img = img.unsqueeze(dim=0)
    return img, src


def inference(src: torch.Tensor, model: nn.Module):
    model.eval()
    with torch.no_grad():
        outputs = model(src)
        # outputs = F.softmax(outputs)
        result = classes[torch.argmax(outputs, dim=1).item()]
    return result



def print_model(
    model_name: str, 
    src: str, 
    weight: str, 
    quantization: str, 
    measure_latency: bool=True, 
    num_classes: int=33,
):
    q = True if quantization is not 'none' else False

    # load model
    if model_name == 'shufflenet':
        from models.shufflenet import ShuffleNetV2
        model = ShuffleNetV2(num_classes=num_classes, pre_trained=False, quantize=q)

    elif model_name == 'mobilenet':
        from models.mobilenet import MobileNetV3
        model = MobileNetV3(num_classes=num_classes, pre_trained=False)

    elif model_name == 'efficientnet':
        from models.efficientnet import EfficientNetV2
        model = EfficientNetV2(num_classes=num_classes, pre_trained=False)

    elif model_name == 'resnet18':
        from models.resnet import resnet18
        model = resnet18(num_classes=num_classes, pre_trained=False, quantize=q)

    elif model_name == 'resnet50':
        from models.resnet import resnet50
        model = resnet50(num_classes=num_classes, pre_trained=False, quantize=q)

    else:
        raise ValueError(f'model name {model_name} does not exists.')
    
    
    # quantization
    if quantization == 'ptq':
        model = ptq_serving(model=model, weight=weight)

    elif quantization == 'qat':
        model = qat_serving(model=model, weight=weight)

    else: # 'none'
        pass

    img, _ = load_image(src)
    result = inference(img, model)
    return result