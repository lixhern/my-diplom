import torch
import os
import random
import glob
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import UNetMulti


def load_model(model_path, device='cuda'):
    model = UNetMulti(n_channels=3, n_classes=2, dropout=0.3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def prepare_tensor(img: Image.Image, size=(256, 256)) -> torch.Tensor:
    img = img.resize(size)
    return transforms.ToTensor()(img)


def predict(model, image_tensor, device='cuda'):
    x = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch.sigmoid(model(x))
    return out.cpu().squeeze(0)


def visualize_prediction(img_tensor, true_liver, true_hema, pred_mask, out_dir, name):
    img_np = img_tensor.permute(1, 2, 0).numpy()
    true_l_np = true_liver.squeeze().numpy()
    true_h_np = true_hema.squeeze().numpy()
    pred_l_np = pred_mask[0].numpy() > 0.5
    pred_h_np = pred_mask[1].numpy() > 0.5

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))
    axs[0].imshow(img_np)
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(true_l_np, cmap='gray')
    axs[1].set_title('True Liver')
    axs[1].axis('off')
    axs[2].imshow(pred_l_np, cmap='gray')
    axs[2].set_title('Pred Liver')
    axs[2].axis('off')
    axs[3].imshow(true_h_np, cmap='gray')
    axs[3].set_title('True Hemangioma')
    axs[3].axis('off')
    axs[4].imshow(pred_h_np, cmap='gray')
    axs[4].set_title('Pred Hemangioma')
    axs[4].axis('off')

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{name}_prediction.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Сохранена визуализация: {save_path}")


def test_model_on_images(model_path, dataset_dir, device='cuda', num_images=3):
    model = load_model(model_path, device)

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    image_paths = []
    for task_id in os.listdir(dataset_dir):
        task_path = os.path.join(dataset_dir, task_id)
        if os.path.isdir(task_path):
            image_paths += glob.glob(os.path.join(task_path, '*.jpg'))

    if not image_paths:
        raise RuntimeError(f"Изображения не найдены в {dataset_dir}")

    random.seed(42)
    num_images = min(num_images, len(image_paths))
    chosen = random.sample(image_paths, num_images)

    out_dir = "visualization"
    for img_path in chosen:
        folder = os.path.dirname(img_path)
        unique_name = os.path.basename(folder)

        liver_mask_path = glob.glob(os.path.join(folder, 'liver_mask.png'))
        hema_mask_path = glob.glob(os.path.join(folder, 'hemangioma_mask.png'))

        if not liver_mask_path or not hema_mask_path:
            print(f"Пропуск {unique_name}: маски не найдены.")
            continue

        img = Image.open(img_path).convert('RGB')
        img_tensor = img_transform(img)
        true_l = prepare_tensor(Image.open(liver_mask_path[0]).convert('L'))
        true_h = prepare_tensor(Image.open(hema_mask_path[0]).convert('L'))

        pred = predict(model, img_tensor, device)

        visualize_prediction(img_tensor, true_l, true_h, pred, out_dir, unique_name)


if __name__ == '__main__':
    model_path = 'unet_best.pth'
    dataset_dir = 'dataset'
    test_model_on_images(model_path, dataset_dir)