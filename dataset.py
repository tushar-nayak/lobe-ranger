import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LungHistPairedDataset(Dataset):
    """
    Custom PyTorch Dataset that pairs 20x and 40x histopathological images
    from the same patient, class, and differentiation level.
    Uses exact columns from data.csv: superclass, subclass, resolution, image_id, patient_id.
    """
    def __init__(self, csv_path, data_dir, patient_ids=None, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load CSV metadata
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Metadata CSV not found at: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        
        # Clean column names
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        
        # Clean text columns
        self.df['superclass'] = self.df['superclass'].astype(str).str.lower().str.strip()
        self.df['resolution'] = self.df['resolution'].astype(str).str.lower().str.strip()
        
        # Handle subclass NaNs cleanly
        self.df['subclass'] = self.df['subclass'].fillna('').astype(str).str.lower().str.strip()
        self.df['subclass'] = self.df['subclass'].replace('nan', '')
        
        # Standardize patient_id and image_id to strings to avoid float/int mismatches
        self.df['patient_id'] = self.df['patient_id'].astype(int)
        
        # Clean image_id (removing any trailing .0 if pandas parsed it as float)
        self.df['image_id'] = self.df['image_id'].astype(str).str.strip()
        self.df['image_id'] = self.df['image_id'].apply(lambda x: x[:-2] if x.endswith('.0') else x)

        # Filter by patient IDs if provided (for patient-wise Train/Val/Test split)
        if patient_ids is not None:
            self.df = self.df[self.df['patient_id'].isin(patient_ids)].reset_index(drop=True)
            
        # Construct exact image filenames and relative paths
        self._construct_image_paths()
        
        # Pair 20x and 40x images patient-wise
        self.pairs = self._pair_images()
        print(f"[{split.upper()}] Paired {len(self.pairs)} image sets across {len(self.df['patient_id'].unique())} patients.")

    def _construct_image_paths(self):
        """Constructs the relative path to every image based on columns"""
        paths = []
        for idx, row in self.df.iterrows():
            sup = row['superclass']
            sub = row['subclass']
            res = row['resolution']
            img_id = row['image_id']
            
            if sup == 'nor' or sup == 'normal':
                filename = f"nor_{res}_{img_id}.jpg"
                dir_name = "nor"
            else:
                filename = f"{sup}_{sub}_{res}_{img_id}.jpg"
                dir_name = f"{sup}_{sub}"
                
            # Full relative path under self.data_dir
            relative_path = os.path.join("images", dir_name, filename)
            paths.append(relative_path)
            
        self.df['image_path'] = paths

    def _pair_images(self):
        pairs = []
        
        # Group by patient_id, superclass, and subclass to ensure perfect matching
        # For normal samples, subclass is empty ('')
        grouped = self.df.groupby(['patient_id', 'superclass', 'subclass'])
        
        for (patient_id, superclass, subclass), group in grouped:
            # Separate into 20x and 40x resolutions
            images_20x = group[group['resolution'].str.contains('20x') | group['resolution'].str.contains('20')]
            images_40x = group[group['resolution'].str.contains('40x') | group['resolution'].str.contains('40')]
            
            list_20x = images_20x['image_path'].tolist()
            list_40x = images_40x['image_path'].tolist()
            
            if not list_20x and not list_40x:
                continue
                
            # Pair 20x and 40x combinations to maximize training sets
            if list_20x and list_40x:
                for img_20 in list_20x:
                    for img_40 in list_40x:
                        pairs.append({
                            'img_20x': img_20,
                            'img_40x': img_40,
                            'patient_id': patient_id,
                            'superclass': superclass,
                            'subclass': subclass
                        })
            elif list_20x:
                # Fallback: if only 20x is available
                for img_20 in list_20x:
                    pairs.append({
                        'img_20x': img_20,
                        'img_40x': img_20,
                        'patient_id': patient_id,
                        'superclass': superclass,
                        'subclass': subclass
                    })
            elif list_40x:
                # Fallback: if only 40x is available
                for img_40 in list_40x:
                    pairs.append({
                        'img_20x': img_40,
                        'img_40x': img_40,
                        'patient_id': patient_id,
                        'superclass': superclass,
                        'subclass': subclass
                    })
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Construct absolute paths
        path_20x = os.path.join(self.data_dir, pair['img_20x'])
        path_40x = os.path.join(self.data_dir, pair['img_40x'])
        
        # Load images
        try:
            img_20x = Image.open(path_20x).convert('RGB')
        except Exception as e:
            # Fallback recursive search if directory nesting shifted
            img_name = os.path.basename(pair['img_20x'])
            img_20x = self._find_and_load_image(img_name)
            
        try:
            img_40x = Image.open(path_40x).convert('RGB')
        except Exception as e:
            img_name = os.path.basename(pair['img_40x'])
            img_40x = self._find_and_load_image(img_name)
            
        # Target A: Malignancy (0 = Normal, 1 = Carcinoma)
        is_malignant = 0 if pair['superclass'] in ['nor', 'normal'] else 1
        
        # Target B: Carcinoma Subtype (0 = Adenocarcinoma/aca, 1 = Squamous Cell/scc)
        subtype = 0
        if pair['superclass'] == 'aca':
            subtype = 0
        elif pair['superclass'] == 'scc':
            subtype = 1
            
        # Target C: Ordinal Differentiation Level
        # Spanish mapping: bd = well (0), md = moderate (1), pd = poor (2)
        # Normal samples are mapped to 0 (and masked out during training anyway)
        diff_map = {
            'bd': 0, 'well': 0,
            'md': 1, 'moderate': 1, 'mod': 1,
            'pd': 2, 'poor': 2
        }
        diff_label = diff_map.get(pair['subclass'], 0)
        
        # Apply transforms
        if self.transform:
            img_20x = self.transform(img_20x)
            img_40x = self.transform(img_40x)
            
        return {
            'image_20x': img_20x,
            'image_40x': img_40x,
            'is_malignant': torch.tensor(is_malignant, dtype=torch.long),
            'subtype': torch.tensor(subtype, dtype=torch.long),
            'differentiation': torch.tensor(diff_label, dtype=torch.long),
            'patient_id': pair['patient_id']
        }

    def _find_and_load_image(self, img_name):
        """Helper to find image if it is inside subdirectories recursively"""
        for root, dirs, files in os.walk(self.data_dir):
            if img_name in files:
                return Image.open(os.path.join(root, img_name)).convert('RGB')
        raise FileNotFoundError(f"Could not find image {img_name} anywhere under {self.data_dir}")

def get_transforms(img_size=224, split='train'):
    """
    Standard training & validation transforms.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
