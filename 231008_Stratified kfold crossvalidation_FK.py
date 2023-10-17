import os
import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging


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


class StratifiedKFoldDatasetSplitter:

    def __init__(self, image_dir, excel_file, n_splits=5, test_size=0.2):
        self.image_dir = image_dir
        self.data = pd.read_excel(excel_file)
        self.n_splits = n_splits
        self.test_size = test_size
        self._setup_dataset()
    
    def _setup_dataset(self):
        unique_study_ids = self._extract_unique_study_ids()
        labels = self._get_labels_for_study_ids(unique_study_ids)

        # Splitting into train+validation and test sets
        ids_train_val, ids_test = train_test_split(
            unique_study_ids,
            test_size=self.test_size,
            stratify=[labels[sid] for sid in unique_study_ids],
            random_state=42
        )
        
        base_save_dir = os.path.join(os.path.dirname(self.image_dir), 'Folded_Dataset')
        test_save_dir = os.path.join(base_save_dir, 'Test')
        
        # Only save test images once, outside the k-fold loop
        self._save_images_by_study_ids(ids_test, test_save_dir)
        
        # Using ids_train_val and labels for the stratified k-fold splitting
        self._perform_stratified_kfold_split(ids_train_val, labels, ids_test)
    
    def _extract_unique_study_ids(self):
        extracted_study_ids = set()
        for _, _, filenames in os.walk(self.image_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    img_name, _ = os.path.splitext(filename)
                    sid = extract_identifier(img_name)  # Modified KK
                    extracted_study_ids.add(sid)
        
        # Keep only study IDs that exist in both the Excel file and the image directory
        valid_study_ids = set(self.data['ID'].unique())
        intersected_study_ids = extracted_study_ids.intersection(valid_study_ids)

        return np.array(list(intersected_study_ids))

    def _get_labels_for_study_ids(self, study_ids):
        labels = {}
        for sid in study_ids:
            label = self.data[self.data['ID'] == sid]['Event'].values[0]
            labels[sid] = label
        return labels

    def _perform_stratified_kfold_split(self, study_ids, labels, ids_test):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=40)
        X = np.array(study_ids)
        y = np.array([labels[sid] for sid in study_ids])
        
        for fold, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
            logging.info(f"Performing split for fold {fold}...")
            
            train_study_ids = X[train_index]
            val_study_ids = X[val_index]
            
            self._save_fold_data(train_study_ids, val_study_ids, fold)
            self._log_fold_statistics(train_study_ids, val_study_ids, ids_test, labels, fold)
        
    def _save_fold_data(self, train_study_ids, val_study_ids, fold):
        base_save_dir = os.path.join(os.path.dirname(self.image_dir), 'Folded_Dataset')
        train_save_dir = os.path.join(base_save_dir, f'Fold_{fold}', 'Train')
        val_save_dir = os.path.join(base_save_dir, f'Fold_{fold}', 'Val')
        
        self._save_images_by_study_ids(train_study_ids, train_save_dir)
        self._save_images_by_study_ids(val_study_ids, val_save_dir)
    
    def _save_images_by_study_ids(self, study_ids, save_dir):
        for root, _, filenames in os.walk(self.image_dir):
            for filename in filenames:
                if filename.endswith('.png'):
                    img_name = os.path.splitext(filename)[0]
                    sid = int(img_name.split('_')[1])
                    if sid in study_ids:
                        relative_path = os.path.relpath(root, self.image_dir)
                        img_save_dir = os.path.join(save_dir, relative_path)
                        os.makedirs(img_save_dir, exist_ok=True)
                        shutil.copy(
                            os.path.join(root, filename),
                            os.path.join(img_save_dir, filename)
                        )
    
    def _log_fold_statistics(self, train_study_ids, val_study_ids, test_study_ids, labels, fold):
        logging.info(f"Fold {fold} Statistics:")
        logging.info("-" * 30)
        
        train_img_count, train_study_count, train_class_dist = self._get_statistics(train_study_ids, labels, fold, 'Train')
        val_img_count, val_study_count, val_class_dist = self._get_statistics(val_study_ids, labels, fold, 'Val')
        
        # Calculating Test statistics
        test_img_count, test_study_count, test_class_dist = self._get_statistics(test_study_ids, labels, fold, 'Test')
        
        logging.info(f"Train: Image patches={train_img_count}, Unique Study ID={train_study_count}, Class Distribution={train_class_dist}")
        logging.info(f"Validation: Image patches={val_img_count}, Unique Study ID={val_study_count}, Class Distribution={val_class_dist}")
        logging.info(f"Test: Image patches={test_img_count}, Unique Study ID={test_study_count}, Class Distribution={test_class_dist}")
        logging.info("\n")
        
        # Writing to Excel
        stats_df = pd.DataFrame({
            'Subset': ['Train', 'Validation', 'Test'],
            'Image_Patches': [train_img_count, val_img_count, test_img_count],
            'Unique_Study_ID': [train_study_count, val_study_count, test_study_count],
            'Study_IDs': [str(list(train_study_ids)), str(list(val_study_ids)), str(list(test_study_ids))],
            'Class_Distribution': [str(train_class_dist.tolist()), str(val_class_dist.tolist()), str(test_class_dist.tolist())]
        })
        stats_df.to_excel(os.path.join(os.path.dirname(self.image_dir), f'Fold_{fold}_statistics.xlsx'), index=False)
    
    def _get_statistics(self, study_ids, labels, fold, subset):
        img_count = sum(len(files) for _, _, files in os.walk(os.path.join(os.path.dirname(self.image_dir), 'Folded_Dataset', f'Fold_{fold}', subset)))
        study_count = len(set(study_ids))
        class_dist = np.bincount([labels[sid] for sid in study_ids])
        return img_count, study_count, class_dist

    def verify_stratification(self):
        all_labels = [label for label in self._get_labels_for_study_ids(self._extract_unique_study_ids()).values()]
        unique_labels = np.unique(all_labels)
        
        overall_distribution = [all_labels.count(label)/len(all_labels) for label in unique_labels]
        
        for fold in range(1, self.n_splits + 1):
            fold_labels = []
            for _, _, filenames in os.walk(os.path.join(self.image_dir, f'Fold_{fold}', 'Train')):
                for filename in filenames:
                    if filename.endswith('.png'):
                        img_name = os.path.splitext(filename)[0]
                        sid = int(img_name.split('_')[1])
                        fold_labels.append(self.data[self.data['ID'] == sid]['Event'].iloc[0])
            
            fold_distribution = [fold_labels.count(label)/len(fold_labels) for label in unique_labels]
            
            plt.figure(figsize=(12, 5))
            plt.bar(unique_labels, overall_distribution, alpha=0.5, label='Overall')
            plt.bar(unique_labels, fold_distribution, alpha=0.5, label=f'Fold_{fold}')
            plt.legend()
            plt.title(f'Distribution comparison: Overall vs Fold_{fold}')
            plt.xlabel('Event')
            plt.ylabel('Proportion')
            plt.xticks(unique_labels)
            plt.show()

# Example usage
image_dir = r'./tiles-TCGA'
excel_file = r'W:/train_val_cohort.xlsx'

stratified_kfold_splitter = StratifiedKFoldDatasetSplitter(image_dir, excel_file, n_splits=5)