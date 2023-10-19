import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.models import densenet121, vgg11
import wandb
import os
from scipy import stats
import optuna
from torch.nn.functional import sigmoid
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
#from torchvision.datasets import DatasetFolder
from torch.utils.data import Dataset
from collections import defaultdict
#This part is added to fix multiprocessing
import multiprocessing
import time
import sys
import config

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        multiprocessing.set_start_method('spawn')

# Addition KK:
def extract_identifier(filename):
    # Attempt extraction using both hyphen and underscore separators
    separators = ['-', '_']
    for separator in separators:
        parts = filename.split(separator)
        if len(parts) >= 3:
            identifier = separator.join(parts[:3])
            return identifier
    return None
#%%
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, df_labels, train=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df_labels = df_labels
        self.df_labels.index = self.df_labels.index.map(extract_identifier)
        self.train = train
        
        # Precompute the list of all image paths
        self.all_images = [os.path.join(subdir, file) 
                           for subdir, dirs, files in os.walk(self.root_dir) 
                           for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))
    	                   and extract_identifier(file) in self.df_labels.index]
        
        if self.train:
            # Perform oversampling

            labels = self.df_labels.loc[[extract_identifier(os.path.basename(file)) for file in self.all_images]]['Event'].to_list()
            class_counts = {label: labels.count(label) for label in set(labels)}
            print(f"Training --> oversampling. Class distribution before oversampling: {class_counts}")

            majority_class = 0
            minority_class = 1

            num_oversample_majority = int(class_counts[majority_class] * 0.20)
            num_oversample_minority = num_oversample_majority + (class_counts[majority_class] - class_counts[minority_class])

            majority_images = [img for img, label in zip(self.all_images, labels) if label == majority_class]
            minority_images = [img for img, label in zip(self.all_images, labels) if label == minority_class]

            oversampled_majority = np.random.choice(majority_images, num_oversample_majority, replace=True)
            oversampled_minority = np.random.choice(minority_images, num_oversample_minority, replace=True)

            self.all_images.extend(oversampled_majority.tolist())
            self.all_images.extend(oversampled_minority.tolist())

            class_counts[majority_class] += num_oversample_majority
            class_counts[minority_class] += num_oversample_minority
            print(f"Class distribution after oversampling: {class_counts}")

        # print(self.all_images[:5])
        # Debug: Print the first few image paths
        print(self.all_images[:5])

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_name = self.all_images[idx]
        
        # Extract study_id from image name
        # segments = img_name.split('_')
        # if len(segments) <= 1:
        #     raise ValueError(f"Unexpected image name format for {img_name}")
        
        study_id = extract_identifier(os.path.basename(img_name))
        
        # Debug: Print the extracted study_id
        # print(f"Extracted study_id: {study_id}")
    
        # Get label from Excel
        # Changed this (KK)
        clin_data = self.df_labels.loc[study_id][['LVI', 'RTI', 'Size']].to_numpy()
        label_entries = [self.df_labels.loc[study_id]['Event']]

        # Debug: Print the label entries from the DataFrame
        #print(f"Label entries for study_id {study_id}: {label_entries}")
        
        if len(label_entries) == 0:
            print(f"No label found for study_id: {study_id} in image {img_name}.")
            label = None
        else:
            label = label_entries[0]
    
        # Load image
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, clin_data, label, study_id


root_dir = "/data/scratch/kkwakkenbos/Tiles_512_10x_normalized"
xlsx_path = "/data/scratch/kkwakkenbos/Cohort_EMC.xlsx"
transform = transforms.Compose([transforms.ToTensor()])

patient_data = pd.read_excel(xlsx_path, engine='openpyxl').set_index('ID').dropna()

if not config.cohort_settings['synchronous']:
    patient_data = patient_data.drop(patient_data[patient_data['Synchronous'] == 1].index)
if not config.cohort_settings['treatment']:
    patient_data = patient_data.drop(patient_data[patient_data['Treatment'] == 1].index)

print(patient_data.head(10))

train_index, val_index, _, _ = train_test_split(
    patient_data.index, patient_data['Event'], stratify=patient_data['Event'], test_size=0.2, random_state=0)


patient_data_train = patient_data.loc[train_index].copy()
patient_data_val = patient_data.loc[val_index].copy()

# Create datasets using CustomImageDataset
train_dataset = CustomImageDataset(os.path.join(root_dir), patient_data_train, train=True, transform=transform)
val_dataset = CustomImageDataset(os.path.join(root_dir), patient_data_val, train=False, transform=transform)

def get_class_counts(dataset):
    class_counts = defaultdict(int)
    for idx, (img, _, label, _) in enumerate(dataset):  # Notice the added underscore
        try:
            class_counts[label] += 1
            if idx == len(dataset) - 1: 
                print(f"Last index processed: {idx}")  # Debugging line
        except Exception as e:
            print(f"Error processing image at index {idx}: {e}")
    return class_counts

train_class_counts = get_class_counts(train_dataset)
val_class_counts = get_class_counts(val_dataset)

print("Training set class counts:")
for label, count in train_class_counts.items():
    print(f"Class {label}: {count} images")

print("\nValidation set class counts:")
for label, count in val_class_counts.items():
    print(f"Class {label}: {count} images")


#%%
def calculate_metrics(y_true, y_pred, y_proba):
    # Assuming y_true and y_pred are already numpy arrays or Python lists. If they are tensors, convert them only once.
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'bal_acc': balanced_accuracy_score(y_true, y_pred),
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0
    }

    for metric, func in [('f1', f1_score), ('precision', precision_score), ('recall', recall_score)]:
        try:
            metrics[metric] = func(y_true, y_pred)
        except:
            pass

    if isinstance(y_proba, torch.Tensor):
        y_proba = y_proba.cpu().numpy()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    metrics['roc_auc'] = auc(fpr, tpr)

    try:
        metrics['cm'] = confusion_matrix(y_true, y_pred)
    except Exception as e:
        print(f"Error computing confusion matrix: {e}")
        metrics['cm'] = None

    return tuple(metrics.values())


def log_metrics(metrics, split, prefix, loss):
    wandb.log({
        f"{prefix}_{split}_loss": loss,
        f"{prefix}_{split}_acc": metrics[0],
        f"{prefix}_{split}_f1": metrics[2],
        f"{prefix}_{split}_balacc": metrics[1],
        f"{prefix}_{split}_recall": metrics[4],
        f"{prefix}_{split}_precision": metrics[3],
        f"{prefix}_{split}_cnfmatrix": metrics[6],
        f"{prefix}_{split}_auc": metrics[5]
        #f"{prefix}_{split}_fpr": metrics[7],
        #f"{prefix}_{split}_tpr": metrics[8]
    })

class CustomHead(nn.Module):
    def __init__(self, input_size, output_size, batch_norm, dropout_rate):
        super(CustomHead, self).__init__()
        # clinical features, fix later
        self.fc = nn.Linear(input_size, output_size)
        
        # Conditionally create the BatchNorm1d layer
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x = torch.cat((x[0], clinical_input), dim=1)

        if self.batch_norm:
            x = self.batch_norm(x)
        
        x = self.fc(x)
        
        # Apply dropout after linear layer (since there's no activation function here)
        x = self.dropout(x)
        
        return x

def create_model(batch_norm, dropout_rate):
    model = densenet121(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier of DenseNet for your task
    num_ftrs = model.classifier.in_features
    model.classifier = CustomHead(num_ftrs, 1, batch_norm, dropout_rate)

    # Making the last few layers trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    model = model.to(device)

    return model, "Densenet121"  # adjust the string if you choose another variant

# Checking for GPU availability
print("Checking for GPU availability...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Gradient accumulation steps
accumulation_steps = 4
# Fixed Number of Epochs
T = 50

def train(model, train_data_loader, optimizer, scheduler, device, scaler):
    start_time = time.time()
    print("[INFO] Starting training...")
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_logits = []

    optimizer.zero_grad()
    print("Debug: Entering the loop...")
    
    start_time = time.time()  # Capturing start time
    
    for i, (images, clin_vars, labels, study_ids) in enumerate(train_data_loader):
       # print(f"Debug: Batch {i}...")
        images, labels = images.to(device), labels.to(device)
        with autocast():
            outputs = model(images)
            #loss_start = time.time()
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            #loss_end = time.time()
            #print(f"Loss calculation in training completed in {loss_end - loss_start:.2f} seconds")
        scaler.scale(loss).backward()

        # Adjusting the optimizer step for every batch
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        torch.cuda.empty_cache()

        all_predictions.extend((sigmoid(outputs.squeeze())>0.5).long().tolist())
        all_labels.extend(labels.tolist())
        y_logits.extend(outputs.squeeze().detach().cpu().numpy())
        total_loss += loss.item() * images.size(0)
        torch.cuda.empty_cache()
        #print(f"[INFO] Batch {i+1}/{len(train_data_loader)} processed.")
    scheduler.step()

    metrics_start_time = time.time()
    metrics = calculate_metrics(all_labels, all_predictions, y_logits)
    metrics_end_time = time.time()

    end_time = time.time()  # Capturing end time
    epoch_time = end_time - start_time  # Calculating time taken for the epoch
    
    print(f"[INFO] Epoch completed in: {epoch_time:.2f} seconds")
    print(f"[INFO] Metrics calculated in: {metrics_end_time - metrics_start_time:.2f} seconds")

    # Predicting for training set for patient-level metrics
    _, _, train_patient_predictions, train_patient_labels, _, train_patient_probs = predict(model, train_data_loader)
    train_patient_metrics = calculate_metrics(train_patient_labels, train_patient_predictions, train_patient_probs)
    del all_predictions, all_labels, y_logits
    end_time = time.time()
    print("[INFO] Training complete.")
    return total_loss / len(train_data_loader.dataset), metrics, train_patient_metrics

def validate(model, val_data_loader, device):
    print("[INFO] Starting validation...")
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    y_proba = []
    
    val_start_time = time.time()

    with torch.no_grad():
        for images, _, labels, study_ids in val_data_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast():
                outputs = model(images)  # changed this line
                
                logits = outputs
                    
                output = logits.view(-1)  # changed this line
            #loss_start = time.time()
            loss = F.binary_cross_entropy_with_logits(outputs.squeeze(), labels.float())
            #loss_end = time.time()
            #print(f"Loss calculation in validation completed in {loss_end - loss_start:.2f} seconds")
            total_loss += loss.item() * images.size(0)

            all_predictions.extend((sigmoid(outputs.squeeze())>0.5).long().tolist())
            all_labels.extend(labels.tolist())
            y_proba.extend(sigmoid(outputs.squeeze().detach().cpu().float()).numpy())

            del images, labels, output
            torch.cuda.empty_cache()  # Freeing up unused memory

    metrics_start_time = time.time()
    metrics = calculate_metrics(all_labels, all_predictions, y_proba)
    metrics_end_time = time.time()
    
    val_end_time = time.time()
    
    print(f"[INFO] Validation completed in: {val_end_time - val_start_time:.2f} seconds")
    print(f"[INFO] Metrics calculated in: {metrics_end_time - metrics_start_time:.2f} seconds")
    
    # Clearing memory
    del all_predictions, all_labels, y_proba

    return total_loss / len(val_data_loader.dataset), metrics

def predict(model, data_loader):
    model.eval()
    model = model.to(device)
    batch_predictions = []
    batch_labels = []
    study_summary = {}
    y_proba = []
    with torch.no_grad():
        for i, (inputs, _, labels, study_ids) in enumerate(data_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            logits = outputs
                
            output = logits.view(-1)  
            probs = sigmoid(output.detach().cpu().float()).numpy()
            y_proba.extend(probs)
            
            batch_predictions.extend((probs > 0.5).astype(int).tolist())
            batch_labels.extend(labels.tolist())
            
            for prediction, probability, label, study_id in zip(batch_predictions, probs, labels.tolist(), study_ids):  # Removed .tolist() here
                if study_id not in study_summary:
                    study_summary[study_id] = {'predictions': [], 'probs': [], 'labels': []}
                study_summary[study_id]['predictions'].append(prediction)
                study_summary[study_id]['probs'].append(probability)
                study_summary[study_id]['labels'].append(label)
        
        patient_predictions = []
        patient_probabilities = []
        patient_labels = []
        for study_id in study_summary:
            patient_predictions.append(stats.mode(study_summary[study_id]['predictions'])[0][0])
            patient_probabilities.append(np.mean(study_summary[study_id]['probs']))
            patient_labels.append(stats.mode(study_summary[study_id]['labels'])[0][0])
        return batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probabilities

if not os.path.exists('Best models'):
    os.makedirs('Best models')

def generate_filename(prefix, run_name, config, criterion, epoch):
    """Generate a filename based on the model configuration."""
    params = [f"{key}={value}" for key, value in config.items() if key not in ['project']]
    return os.path.join("Best models", f"{prefix}_{run_name}_epoch_{epoch + 1}_{'_'.join(params)}_{criterion}.pt")

def objective(trial):
    run = wandb.init(project="230919_brs(1&2)VS.3_DensNet121_BINCROSENT_oversampl_augm_Macenkonorm_10lyrstrainable_FK.py", config={})
    wandb.save(".\Works-well\230919_brs(1&2)VS.3_DensNet121_BINCROSENT_oversampl_augm_Macenkonorm_10lyrstrainable_FK.py")
    
    config = run.config
    trial_lr = trial.suggest_categorical('lr', [1e-4, 1e-3, 1e-2])
    trial_batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    trial_dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2, 0.5])

    # Configuration Setup
    run.config.update({
        "lr": trial_lr,
        "batch_size": trial.suggest_categorical('batch_size', [8]),
        "num_epochs": T,
        "weight_decay": trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True),
        "step_size": 5,
        "gamma": trial.suggest_float('gamma', 0.1, 0.9, step=0.1),
        "batch_norm": trial_batch_norm,
        "dropout_rate": trial_dropout_rate
    }, allow_val_change=True)

    # Create DataLoaders
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=1, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    
    # Initialize the model, get the model name and update the run configuration with the model name
    model, model_name = create_model(config["batch_norm"], config["dropout_rate"])
    config["model_name"] = model_name  # Update the config with the dynamic model name
    run.config.update(config, allow_val_change=True)

    # Move the model to the device
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_bal_acc = float('-inf')
    early_stop_counter = 0
    early_stop_limit = 15
    
    for epoch in range(T):
        training_loss, training_metrics, train_patient_metrics = train(model, train_data_loader, optimizer, scheduler, device, scaler)
        print(f"Epoch {epoch + 1} - Training Loss: {training_loss}")
        log_metrics(training_metrics, 'train', 'img', training_loss)
        log_metrics(train_patient_metrics, 'train', 'ptnt', None)

        validation_loss, validation_metrics = validate(model, val_data_loader, device)
        print(f"Epoch {epoch + 1} - Validation Loss: {validation_loss}")
        
        bal_acc = validation_metrics[2]

        if validation_loss < best_val_loss:
            best_val_loss = validation_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), generate_filename('best_model', run.name, config, 'loss', epoch))
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_limit:
                print('Early stopping')
                break
    
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            torch.save(model.state_dict(), generate_filename('best_model', run.name, config, 'bal_acc', epoch))

        trial.report(validation_loss, epoch)
        log_metrics(validation_metrics, 'val', 'img', validation_loss)
        print(f'Image-Level - Epoch {epoch + 1} - Accuracy: {validation_metrics[0]}, F1 Score: {validation_metrics[2]}, Balanced Accuracy: {validation_metrics[1]}')
        
        batch_predictions, batch_labels, patient_predictions, patient_labels, y_proba, patient_probs = predict(model, val_data_loader)
        
        patient_metrics = calculate_metrics(patient_labels, patient_predictions, patient_probs)
        log_metrics(patient_metrics, 'val', 'ptnt', None)
        print(f'Patient-Level - Epoch {epoch + 1} - Accuracy: {patient_metrics[0]}, F1 Score: {patient_metrics[2]}, Balanced Accuracy: {patient_metrics[1]}')
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    run.finish()
    return best_val_loss
#Chagned this part for multiprocessing
if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=200)