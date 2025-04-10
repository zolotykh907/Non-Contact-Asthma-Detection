import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

def fix_negative_keypoints(keypoints, limb_connections):
    num_keypoints = keypoints.shape[0]
    fixed_keypoints = keypoints.copy()

    # Находим все видимые точки (координаты >= 0)
    visible_indices = np.where((fixed_keypoints[:, 0] >= 0) & (fixed_keypoints[:, 1] >= 0))[0]

    for i in range(num_keypoints):
        if fixed_keypoints[i, 0] < 0 or fixed_keypoints[i, 1] < 0:
            # Находим все связанные точки
            connected_points = []
            for pair in limb_connections:
                if i in pair:
                    connected_point_idx = pair[0] if pair[1] == i else pair[1]
                    if fixed_keypoints[connected_point_idx, 0] >= 0 and fixed_keypoints[connected_point_idx, 1] >= 0:
                        connected_points.append(connected_point_idx)

            if connected_points:
                # Вычисляем среднее значение координат
                avg_x = np.mean([fixed_keypoints[j, 0] for j in connected_points])
                avg_y = np.mean([fixed_keypoints[j, 1] for j in connected_points])
                fixed_keypoints[i, 0] = avg_x
                fixed_keypoints[i, 1] = avg_y
            else:
                # Если связанные точки также отрицательные, находим ближайшую видимую точку
                if len(visible_indices) > 0:
                    # Вычисляем расстояния до всех видимых точек
                    distances = np.linalg.norm(fixed_keypoints[visible_indices, :2] - fixed_keypoints[i, :2], axis=1)
                    nearest_idx = visible_indices[np.argmin(distances)]
                    fixed_keypoints[i, :2] = fixed_keypoints[nearest_idx, :2]
                else:
                    # Если видимых точек нет, используем центр изображения (например, [0.5, 0.5])
                    fixed_keypoints[i, :2] = [0.5, 0.5]

    return fixed_keypoints

# Определение датасета
# Определение датасета
class DepthKeypointDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, limb_connections=None):
        self.keypoints_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.limb_connections = limb_connections

    def __len__(self):
        return len(self.keypoints_df)

    def __getitem__(self, idx):
        img_name = self.keypoints_df.iloc[idx, 0]  # Первый столбец — имя файла
        img_path = os.path.join(self.img_dir, img_name)
        depth_image = Image.open(img_path).convert('L')  # Загружаем как grayscale

        # Преобразуем изображение в тензор
        if self.transform:
            depth_image = self.transform(depth_image)

        # Загружаем ключевые точки (33 точки * 4 значения: x, y, z, видимость)
        keypoints = self.keypoints_df.iloc[idx, 1:].values.astype('float32')  # Все столбцы, кроме первого
        keypoints = keypoints.reshape(-1, 4)  # Преобразуем в [33, 4]

        # Преобразуем нормализованные координаты в абсолютные
        img_width, img_height = depth_image.shape[2], depth_image.shape[1]  # Получаем размеры тензора
        keypoints[:, 0] = (keypoints[:, 0]) * img_width
        keypoints[:, 1] = (keypoints[:, 1]) * img_height

        # Исправляем отрицательные координаты
        if self.limb_connections:
            keypoints = fix_negative_keypoints(keypoints, self.limb_connections)

        return depth_image, keypoints

# Определение модели (упрощённая версия)
class SimplePoseModel(nn.Module):
    def __init__(self, num_keypoints=33):
        super(SimplePoseModel, self).__init__()
        self.num_keypoints = num_keypoints

        # Backbone (основная CNN)
        self.backbone = nn.Sequential(
            # Первый блок
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Второй блок
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Третий блок
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Четвёртый блок
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Пятый блок
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Head для предсказания координат ключевых точек
        self.fc = nn.Sequential(
            nn.Linear(1024 * 7 * 7, 2048),  # После MaxPool2d размер изображения 7x7
            nn.ReLU(),
            nn.Dropout(0.3),  # Добавляем Dropout для регуляризации
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_keypoints * 2),  # Предсказываем x и y для каждой ключевой точки
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Преобразуем в вектор
        x = self.fc(x)
        x = x.view(x.size(0), self.num_keypoints, 2)  # Преобразуем в [batch_size, num_keypoints, 2]
        return x

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Нормализация для grayscale
])

limb_connections = [
    (0, 1),   # Нос -> Внутренний угол левого глаза
    (1, 2),   # Внутренний угол левого глаза -> Левый глаз
    (2, 3),   # Левый глаз -> Внешний угол левого глаза
    (0, 4),   # Нос -> Внутренний угол правого глаза
    (4, 5),   # Внутренний угол правого глаза -> Правый глаз
    (0, 9),   # Нос -> Левый рот
    (0, 10),  # Нос -> Правый рот
    (9, 10),  # Левый рот -> Правый рот
    (11, 13), # Левое плечо -> Левый локоть
    (13, 15), # Левый локоть -> Левое запястье
    (15, 17), # Левое запястье -> Левый мизинец
    (15, 19), # Левое запястье -> Левый указательный палец
    (19, 21), # Левый указательный палец -> Кончик левого указательного пальца
    (15, 16), # Левое запястье -> Левый большой палец
    (16, 18), # Левый большой палец -> Кончик левого большого пальца
    (12, 14), # Правое плечо -> Правый локоть
    (14, 16), # Правый локоть -> Правое запястье
    (16, 20), # Правое запястье -> Правый мизинец
    (16, 22), # Правое запястье -> Правый указательный палец
    (22, 24), # Правый указательный палец -> Кончик правого указательного пальца
    (16, 18), # Правое запястье -> Правый большой палец
    (18, 20), # Правый большой палец -> Кончик правого большого пальца
    (11, 23), # Левое плечо -> Левое бедро
    (23, 25), # Левое бедро -> Левое колено
    (25, 27), # Левое колено -> Левая лодыжка
    (27, 29), # Левая лодыжка -> Левый большой палец ноги
    (27, 31), # Левая лодыжка -> Левый мизинец ноги
    (12, 24), # Правое плечо -> Правое бедро
    (24, 26), # Правое бедро -> Правое колено
    (26, 28), # Правое колено -> Правая лодыжка
    (28, 30), # Правая лодыжка -> Правый большой палец ноги
    (28, 32), # Правая лодыжка -> Правый мизинец ноги
]

# Загрузка данных
csv_file = "/var/data/pose_estimation/Diploma/csv_depth/26_merged.csv"  # Путь к CSV-файлу
img_dir = "/var/data/pose_estimation/Diploma/26_combined_images"  # Папка с глубинными изображениями
output_image_dir = "/var/data/pose_estimation/Diploma/26_saved_img4"
dataset = DepthKeypointDataset(csv_file, img_dir, transform=transform, limb_connections=limb_connections)

# img_width, img_height = 640, 480

# Разделение на обучающую и валидационную выборки
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True)

# Создание DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
# чисто ищем видеоядро
# Инициализация модели, функции потерь и оптимизатора
model = SimplePoseModel(num_keypoints=33).to(device)
criterion = nn.SmoothL1Loss().to(device)  # Функция потерь для координат ключевых точек
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

# Функция для вычисления метрик
def calculate_metrics(predictions, ground_truth, image_size):
    """
    Вычисляет метрики: PCK, NME, Cosine Similarity, Keypoints Accuracy, MPE.
    :param predictions: Предсказанные ключевые точки [batch_size, num_keypoints, 2].
    :param ground_truth: Истинные ключевые точки [batch_size, num_keypoints, 4].
    :param image_size: Размер изображения (ширина, высота).
    :return: Словарь с метриками.
    """
    batch_size, num_keypoints, _ = predictions.shape

    # Вычисление метрик
    image_size_tensor = torch.tensor(image_size, dtype=torch.float32)
    image_diagonal = torch.sqrt(image_size_tensor[0] ** 2 + image_size_tensor[1] ** 2)

    # PCK (Percentage of Correct Keypoints)
    distances = torch.norm(predictions - ground_truth[:, :, :2], dim=2)  # [batch_size, num_keypoints]
    threshold = 0.03 * image_diagonal  # Порог 5% от диагонали изображения
    pck = torch.mean((distances < threshold).float()).item() * 100

    # NME (Normalized Mean Error)
    nme = torch.mean(distances / image_diagonal).item()

    # Cosine Similarity
    pred_flat = predictions.view(-1, 2)  # [batch_size * num_keypoints, 2]
    gt_flat = ground_truth[:, :, :2].view(-1, 2)  # [batch_size * num_keypoints, 2]
    cos_sim = F.cosine_similarity(pred_flat, gt_flat, dim=1).mean().item()

    # Keypoints Accuracy (с использованием экспоненциального преобразования)
    keypoints_acc = torch.mean(distances).item()

    # keypoints_acc = torch.exp(-mean_distance).item()

    # MPE (Mean Percentage Error)
    mpe = torch.mean(torch.abs((predictions - ground_truth[:, :, :2]) / (ground_truth[:, :, :2] + 1e-6))).item() * 100

    return {
        "PCK": pck,
        "NME": nme,
        "Cosine Similarity": cos_sim,
        "Keypoints Accuracy": keypoints_acc,
        "MPE": mpe,
    }

# Функция для визуализации предсказаний
def visualize_keypoints(images, keypoints, predictions, output_dir, prefix="keypoints"):
    """
    Визуализация ключевых точек на изображении.
    :param images: Изображения [batch_size, 1, height, width].
    :param keypoints: Истинные ключевые точки [batch_size, num_keypoints, 4].
    :param predictions: Предсказанные ключевые точки [batch_size, num_keypoints, 2].
    :param output_dir: Директория для сохранения изображений.
    :param prefix: Префикс для имени файла.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(images.size(0)):
        img = images[i].squeeze().cpu().numpy()  # [height, width]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        kp = keypoints[i].cpu().numpy()  # [num_keypoints, 4]
        pred = predictions[i].cpu().numpy()  # [num_keypoints, 2]

        for j in range(kp.shape[0]):
            if kp[j, 3] > 0.5:  # Если точка видима
                cv2.circle(img, (int(kp[j, 0]), int(kp[j, 1])), 3, (0, 255, 0), -1)  # Истинные точки (зеленый)
                cv2.circle(img, (int(pred[j, 0]), int(pred[j, 1])), 3, (255, 0, 0), -1)  # Предсказанные точки (синий)

        img_path = os.path.join(output_dir, f"{prefix}_image_{i+1}.png")
        cv2.imwrite(img_path, img)


# Обучение модели
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, keypoints in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device=1)
        keypoints = keypoints.to(device=1)
        # # Отладочный вывод
        # print("Keypoints min:", keypoints.min().item(), "max:", keypoints.max().item())
        # print("Predictions min:", predictions.min().item(), "max:", predictions.max().item())

        optimizer.zero_grad()
        predictions = model(images)

        # Масштабируем предсказания обратно в абсолютные координаты
        # img_width, img_height = images.size(-1), images.size(-2)
        # print(img_width, img_height)
        # predictions = predictions * torch.tensor([img_width, img_height], device=predictions.device)
        # print(predictions, keypoints)

        # Вычисление потерь
        loss = criterion(predictions, keypoints[:, :, :2])  # Используем только x и y
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # train_loss = loss.item()


    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}")
    # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")


    # Валидация
    model.eval()
    val_loss = 0.0
    metrics = []
    with torch.no_grad():
        for images, keypoints in tqdm(val_loader, desc=f"Validation"):
            images = images.to(device=1)
            keypoints = keypoints.to(device=1)
            predictions = model(images)

            # Преобразуем предсказанные координаты в абсолютные
            # img_width, img_height = images.size(-1), images.size(-2)
            # predictions = predictions / torch.tensor([img_width, img_height], device=predictions.device)

            # Вычисление потерь
            loss = criterion(predictions, keypoints[:, :, :2])
            val_loss += loss.item()

            # Вычисление метрик
            batch_image_size = (images.size(-2), images.size(-1))
            metrics.append(calculate_metrics(predictions, keypoints, batch_image_size))

            # Визуализация предсказаний
            visualize_keypoints(images, keypoints, predictions, output_image_dir, prefix=f"epoch_{epoch+1}")

    val_loss /= len(val_loader)
    mean_metrics = {
        "PCK": np.mean([m["PCK"] for m in metrics]),
        "NME": np.mean([m["NME"] for m in metrics]),
        "Cosine Similarity": np.mean([m["Cosine Similarity"] for m in metrics]),
        "Keypoints Accuracy": np.mean([m["Keypoints Accuracy"] for m in metrics]),
        "MPE": np.mean([m["MPE"] for m in metrics]),
    }

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Metrics -> PCK: {mean_metrics['PCK']:.2f}%, NME: {mean_metrics['NME']:.4f}, "
          f"Cosine Similarity: {mean_metrics['Cosine Similarity']:.4f}, "
          f"Keypoints Accuracy: {mean_metrics['Keypoints Accuracy']:.2f}%, "
          f"MPE: {mean_metrics['MPE']:.4f}")

# Сохранение модели
torch.save(model.state_dict(), "26simple_pose_model4.pth")