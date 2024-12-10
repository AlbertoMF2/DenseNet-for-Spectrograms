import os
import subprocess
import time
import re
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn  # Módulos usados en las redes neuronales: capas, funciones de pérdida, normalización, etc.
import torch.nn.functional as F
import torch.optim as optim  # Algoritmos de optimización
from torch.utils.data import Dataset, DataLoader  # Facilita el manejo de datasets

from torchvision import transforms, models
from torch_lr_finder import LRFinder

start_time = time.time()

print(os.path.basename(__file__))

# ======================================================================================================================
# ======================================================================================================================

def get_gpu_memory_usage():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    memory_usage = [int(x) for x in output.decode('utf-8').strip().split('\n')]
    return memory_usage

def get_gpu_with_least_memory():
    memory_usage = get_gpu_memory_usage()
    return memory_usage.index(min(memory_usage))

gpu_index = get_gpu_with_least_memory()
print(f"Using GPU {gpu_index} with the least memory usage.")
#Configurar el dispositivo
device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

# ======================================================================================================================
# ======================================================================================================================

# Hyperparameters
learning_rate = 0.0002 #0.0005. LO BAJO PARA MAYOR SUAVIDAD
batch_size = 32 #64. LO BAJO PARA MENOR SOBREAJUSTE
num_epochs = 25 #100. LO BAJO PQ NO MEJORABA A PARTIR DE CIERTAS ÉPOCAS
weight_decay = 1e-3 #0.1 #0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
delta = 1.0

# ======================================================================================================================
# ======================================================================================================================

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train_image_files = []
        self.valid_image_files = []
        
        pattern = re.compile(r'spectrogram_no_patologia_ecg_normal_no_repe(\d+)_canal11\b')
        for subdir, _, files in os.walk(root_dir):
            subfolder_name = os.path.basename(subdir)
            match = pattern.match(subfolder_name)
            if match:
                folder_num = int(match.group(1))
                for filename in files:
                    if filename.endswith(".tiff") and os.path.isfile(os.path.join(subdir, filename)):
                        file_path = os.path.join(subdir, filename)
                        if 0 <= folder_num <= 11:
                            self.train_image_files.append(file_path)
                        elif 12 <= folder_num <= 13:
                            self.valid_image_files.append(file_path)
        
        print(f"Total de imágenes de entrenamiento: {len(self.train_image_files)}")
        print(f"Total de imágenes de validación: {len(self.valid_image_files)}")
        
        self.train_targets = [self.extract_age(filename) for filename in self.train_image_files]
        self.valid_targets = [self.extract_age(filename) for filename in self.valid_image_files]
        
        if self.transform is None:
            self.calculate_global_stats()
            
        # Variable para controlar si ya hemos impreso el primer archivo
        self.first_file_printed = False


    def __len__(self):
        return len(self.train_image_files) + len(self.valid_image_files)

    def __getitem__(self, idx):
        if idx < len(self.train_image_files):
            img_path = self.train_image_files[idx]
            target = self.train_targets[idx]
        else:
            img_path = self.valid_image_files[idx - len(self.train_image_files)]
            target = self.valid_targets[idx - len(self.train_image_files)]
        
        image = self.load_and_preprocess(img_path)
        
        if self.transform:
            image = self.transform(image)
        
        img_name = os.path.basename(img_path)
        '''
        # Imprimir el nombre del primer archivo solo una vez
        if not self.first_file_printed:
            print(f"Primer archivo de imagen cargado: {img_name}")
            self.first_file_printed = True
        '''
        return image, target, img_name

    def extract_age(self, filename):
        return float(os.path.basename(filename).split('_')[3])

    def load_and_preprocess(self, img_path):
        image = Image.open(img_path)
        spectrogram = np.array(image).astype(np.float32)
        log_spectrogram = np.log1p(spectrogram)  # log1p is log(1+x), avoiding log(0)
        return log_spectrogram

    def calculate_global_stats(self):
        all_spectrograms = [self.load_and_preprocess(file) for file in self.train_image_files + self.valid_image_files]
        all_spectrograms = np.stack(all_spectrograms)
        
        global_mean = np.mean(all_spectrograms)
        global_std = np.std(all_spectrograms)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[global_mean], std=[global_std])
        ])

# Usage
dataset = CustomDataset(root_dir="/datos/work/ECG/CODE15/espectrogramas")
train_dataset = torch.utils.data.Subset(dataset, range(len(dataset.train_image_files)))
valid_dataset = torch.utils.data.Subset(dataset, range(len(dataset.train_image_files), len(dataset)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(f"Size of train_loader: {len(train_loader.dataset)}")
print(f"Size of valid_loader: {len(valid_loader.dataset)}")

# ======================================================================================================================
# ======================================================================================================================


def modify_densenet_for_regression(model):
    model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Get the number of features in the last layer
    num_features = model.classifier.in_features
    
    # Create a new classifier
    new_classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    # Replace the classifier
    model.classifier = new_classifier
    
    return model

# Load the pre-trained DenseNet121 model
densenet121 = models.densenet121(weights='DEFAULT')

# Modify the model for regression
densenet121 = modify_densenet_for_regression(densenet121)

# ======================================================================================================================
# ======================================================================================================================

densenet121.to(device)

# ======================================================================================================================
# ======================================================================================================================

# Define la función de pérdida
criterion = nn.HuberLoss(delta=delta)

# Define el optimizador AdamW
optimizer = optim.AdamW(densenet121.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)#, threshold=0.0001)#, min_lr=1e-6)

# ======================================================================================================================
# ======================================================================================================================

train_losses = []  # Lista para almacenar las pérdidas de entrenamiento
valid_losses = []

scaler = torch.cuda.amp.GradScaler()

best_train_preds = []
best_train_targs = []
best_train_files = []
best_valid_preds = []
best_valid_targs = []
best_valid_files = []

best_valid_loss = float('inf')
best_model_path = f'{os.path.basename(__file__)}_best_model.pth'

for epoch in range(num_epochs):
    epoch_train_loss = 0.0
    epoch_valid_loss = 0.0

    # ENTRENAMIENTO
    
    densenet121.train()

    # Listas para ver alguna predicción y su target real
    train_preds = []
    train_targs = []
    train_files = []

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for batch_idx, (data, targets, img_names) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.unsqueeze(1).float().to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # Forward pass
                scores = 10 + 85*densenet121(data)
                loss = criterion(scores, targets)

            scaler.scale(loss).backward()
            
            torch.nn.utils.clip_grad_norm_(densenet121.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()

            # Update the training progress bar
            pbar.set_postfix({'Train Loss': epoch_train_loss / (batch_idx + 1)}, update=True)  # Update the message in the progress bar
            pbar.update(1)  # Increment the progress counter

            train_preds.extend(scores.detach().cpu().numpy())
            train_targs.extend(targets.cpu().numpy())
            train_files.extend(img_names)
            
            # Calcular la media entre las predicciones y los targets reales para entrenamiento
        train_mean_diff = np.mean(np.abs(np.array(train_preds) - np.array(train_targs)))

    # VALIDACIÓN
    densenet121.eval()

    # Listas para ver alguna predicción y su target real
    valid_preds = []
    valid_targs = []
    valid_files = []
    
    with torch.no_grad():
        for data, targets, img_names in valid_loader:
            data = data.to(device=device)
            targets = targets.unsqueeze(1).float().to(device)

            # Forward pass
            scores = 10 + 85*densenet121(data)
            loss = criterion(scores, targets)

            epoch_valid_loss += loss.item()

            valid_preds.extend(scores.cpu().numpy())
            valid_targs.extend(targets.cpu().numpy())
            valid_files.extend(img_names)
        
    # Calcular la media entre las predicciones y los targets reales para validación
    valid_mean_diff = np.mean(np.abs(np.array(valid_preds) - np.array(valid_targs)))
        
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Mean Diff: {train_mean_diff:.2f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Valid Mean Diff: {valid_mean_diff:.2f}")
        
    # Almacena la pérdida de entrenamiento y de validación de la época actual
    train_losses.append(epoch_train_loss / len(train_loader))
    valid_losses.append(epoch_valid_loss / len(valid_loader))

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}')
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {valid_losses[-1]:.4f}')
    
    scheduler.step(valid_losses[-1])
    
     # Guardar el mejor modelo
    if epoch_valid_loss / len(valid_loader) < best_valid_loss:
        best_valid_loss = epoch_valid_loss / len(valid_loader)
        torch.save(densenet121.state_dict(), best_model_path)
        print(f"New best model saved with validation loss: {best_valid_loss}")
        
        # Guardar el nombre de la imagen, las predicciones y los targets de la mejor época
        best_train_preds = train_preds
        best_train_targs = train_targs
        best_train_files = train_files
        best_valid_preds = valid_preds
        best_valid_targs = valid_targs
        best_valid_files = valid_files

densenet121.load_state_dict(torch.load(best_model_path))

# CÁLCULO DE PARÁMETROS
def evaluate_model(loader, model, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    true_ages = []
    predicted_ages = []

    with torch.no_grad():
        for data, targets, img_names in loader:
            data = data.to(device)
            targets = targets.unsqueeze(1).float().to(device)
            true_ages.extend(targets.cpu().detach().numpy())

            scores = 10 + 85 * model(data)
            loss = criterion(scores, targets)
            
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            predicted_ages.extend(scores.cpu().numpy().flatten())
     
    total_loss /= total_samples
     
    true_ages = np.array(true_ages).squeeze()
    predicted_ages = np.array(predicted_ages).squeeze()
        
    mae = mean_absolute_error(true_ages, predicted_ages)
    mse = mean_squared_error(true_ages, predicted_ages)
    r, _ = pearsonr(true_ages, predicted_ages)
    R2 = r2_score(true_ages, predicted_ages)

    return total_loss, mae, mse, r, R2, predicted_ages, true_ages

train_loss, train_mae, train_mse, train_r, train_R2, train_predictions, train_targets  = evaluate_model(train_loader, densenet121, criterion, device)
valid_loss, valid_mae, valid_mse, valid_r, valid_R2, valid_predictions, valid_targets  = evaluate_model(valid_loader, densenet121, criterion, device)

print(f"Error on training set: {train_loss:.4f}")
print(f"Error on valid set: {valid_loss:.4f}")
print(f"MAE on training set: {train_mae:.4f}")
print(f"MAE on valid set: {valid_mae:.4f}")
print(f"MSE on training set: {train_mse:.4f}")
print(f"MSE on valid set: {valid_mse:.4f}")
print(f"Correlation coefficient (r) on training set: {train_r:.4f}")
print(f"Correlation coefficient (r) on valid set: {valid_r:.4f}")
print(f"Coefficient of determination (R^2) on training set: {train_R2:.4f}")
print(f"Coefficient of determination (R^2) on valid set: {valid_R2:.4f}")

train_data = pd.DataFrame({
    'File': best_train_files,
    'Predictions': np.array(best_train_preds).flatten(),
    'Targets': np.array(best_train_targs).flatten()
})

valid_data = pd.DataFrame({
    'File': best_valid_files,
    'Predictions': np.array(best_valid_preds).flatten(),
    'Targets': np.array(best_valid_targs).flatten()
})

# Guardar los resultados en CSV
train_data.to_csv(f'{os.path.basename(__file__)}_best_train_predictions_vs_targets.csv', index=False)
valid_data.to_csv(f'{os.path.basename(__file__)}_best_valid_predictions_vs_targets.csv', index=False)

train_data = train_data.sample(frac=0.01)
valid_data = valid_data.sample(frac=0.01)

# Crear la gráfica de regresión para el conjunto de entrenamiento
sns.regplot(x='Predictions', y='Targets', data=train_data)

# Añadir etiquetas y título
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.title('Regression Plot - Training Set')

plt.savefig(f'{os.path.basename(__file__)}_RegressionTrain_lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}.svg')

# Crear la gráfica de regresión para el conjunto de prueba
sns.regplot(x='Predictions', y='Targets', data=valid_data)

# Añadir etiquetas y título
plt.xlabel('Predictions')
plt.ylabel('Targets')
plt.title('Regression Plot - Test Set')

# Mostrar la gráfica
plt.savefig(f'{os.path.basename(__file__)}_RegressionValid_lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}.svg')

plt.clf()

# Plot the losses
epochs = list(range(1, num_epochs + 1))  # Creamos una lista de épocas comenzando desde 1
plt.plot(epochs, train_losses, label='Training Loss')  # Usamos train_losses en lugar de train_loss
plt.plot(epochs, valid_losses, label='Validation Loss')  # Replicamos test_loss a lo largo de las épocas
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.xticks(range(1, num_epochs + 1))  # Eje de épocas solo en enteros
plt.savefig(f'{os.path.basename(__file__)}_lr_{learning_rate}_bs_{batch_size}_epochs_{num_epochs}.svg')

# =======================================================================================================================
# =======================================================================================================================

end_time = time.time()
elapsed_time = end_time - start_time
print("Tiempo de ejecución:", elapsed_time, "segundos")
