import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class CustomCSVDataset(Dataset):
    def __init__(self, file_path, target_column=None, 
                 numeric_cols=None, categorical_cols=None, binary_cols=None,
                 transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        self.target_column = target_column
        
        if numeric_cols is None:
            numeric_cols = self.data.select_dtypes(include=np.number).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
        
        if categorical_cols is None:
            categorical_cols = self.data.select_dtypes(include='object').columns.tolist()
            if target_column in categorical_cols:
                categorical_cols.remove(target_column)
        
        if binary_cols is None:
            binary_cols = self.data.select_dtypes(include='bool').columns.tolist()
            if target_column in binary_cols:
                binary_cols.remove(target_column)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('bin', 'passthrough', binary_cols)
            ])
        
        features = self.data.drop(columns=[target_column] if target_column else [])
        self.features = self.preprocessor.fit_transform(features)
        
        if target_column:
            self.targets = self.data[target_column].values
        else:
            self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]
        if isinstance(features, np.ndarray):
            features = features.astype(np.float32)

        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        if self.targets is not None:
            target = self.targets[idx]
            if isinstance(target, str) or (hasattr(target, 'dtype') and target.dtype == object):
                target = self.label_encoder.transform([target])[0]
            target_tensor = torch.tensor(target, dtype=torch.long)
            return features_tensor, target_tensor
        
        if self.transform:
            features_tensor = self.transform(features_tensor)
            
        return features_tensor

    def get_feature_names(self):
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                cats = transformer.get_feature_names_out(columns)
                feature_names.extend(cats)
            elif name == 'bin':
                feature_names.extend(columns)
        return feature_names
    

dataset = CustomCSVDataset(
    file_path='data.csv',
    target_column='label',
    numeric_cols=['age', 'income'],
    categorical_cols=['city', 'gender'],
    binary_cols=['is_student']
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for features, labels in dataloader:
    pass