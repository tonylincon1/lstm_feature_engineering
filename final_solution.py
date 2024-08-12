import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from catboost import CatBoostClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from imbens.ensemble import AsymBoostClassifier


# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin

from pathlib import Path
import pickle
from os import PathLike
import torch
from tqdm import tqdm

class ThresholdReplacer(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):
    
    def __init__(self, threshold: int, unk_value: int = -1) -> None:
        super().__init__()
        self.threshold = threshold
        self.unk_value = unk_value

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        self.columnwise_keep_categories = {
            column: self._get_keep_categories(X[column]) for column in X.columns
        }

        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        out = X.copy()

        for column, keep_categories in self.columnwise_keep_categories.items():
          unique_values = out[column].unique()
          to_replace = set(unique_values).difference(set(keep_categories))
          out[column].replace(to_replace, self.unk_value, inplace=True)

        return out

    def _get_keep_categories(self, series: pd.Series) -> pd.Series:
        return ( 
            series
              .value_counts()
              .index[(series.value_counts() > self.threshold)]
        ) # type: ignore
    
class DREncoder(torch.nn.Module):

    def __init__(self, 
                 latent_dim: int=16, 
                 geo_lv1_size: int=31, 
                 geo_lv2_size: int=1414,
                 geo_lv3_size: int=11861) -> None:
        super().__init__()
        self.geo_level1_embeddings = torch.nn.Embedding(geo_lv1_size, 16)
        self.geo_level2_embeddings = torch.nn.Embedding(geo_lv2_size, 128)
        self.geo_level3_embeddings = torch.nn.Embedding(geo_lv3_size, 128) 
        self.compressor = torch.nn.Linear(16+128+128, latent_dim)

    def forward(self, x):
        x_1 = self.geo_level1_embeddings(x[:, 0])
        x_2 = self.geo_level2_embeddings(x[:, 1])
        x_3 = self.geo_level3_embeddings(x[:, 2])
        x = torch.concat((x_1, x_2, x_3), dim=1)
        x = torch.nn.functional.relu(x)
        return torch.nn.functional.relu(self.compressor(x))


class GeoDimensionReduction0(BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin):

    def __init__(
            self,
            device,
            path: PathLike,
            latent_dim: int=16, 
            geo_lv1_size: int=31,
            geo_lv2_size: int=1418,
            geo_lv3_size: int=11861) -> None:
        super().__init__()
        self.path = path
        self.device = device
        self.model = DREncoder(
            latent_dim, 
            geo_lv1_size,
            geo_lv2_size,
            geo_lv3_size
        )
        self.latent_dim = latent_dim
        self.geo_lv1_size = geo_lv1_size
        self.geo_lv2_size = geo_lv2_size
        self.geo_lv3_size = geo_lv3_size
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def fit(self, X: pd.DataFrame, y=None, *args, **kwargs):
        return self

    def transform(self, X: pd.DataFrame, y=None, *args, **kwargs):
        # Convert pd to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values # type: ignore
        # Apply encoder
        self.model.eval()
        X = torch.from_numpy(X).type(torch.long) # type: ignore
        return self.model(X).detach().numpy()
    
def create_submission(predictions, submission_formats_path: PathLike):
    submission_format = pd.read_csv(submission_formats_path, index_col=0)
    submission = pd.DataFrame(data=predictions, columns=submission_format.columns, index=submission_format.index)
    submission['damage_grade'] = submission['damage_grade'].astype(int)
    return submission

def train_algorithm(random_seed):
    BASE_DIR = Path.cwd()
    desired_dir = BASE_DIR
    while desired_dir.name != 'api_richter_predictor' and desired_dir != desired_dir.parent:
        desired_dir = desired_dir.parent
    BASE_DIR = desired_dir
    print(BASE_DIR)
    DATA_DIR = BASE_DIR / 'data' / 'raw'
    MODEL_DIR = BASE_DIR / 'models'
    SUBMISSION_DIR = BASE_DIR / 'submissions'

    TRAINING_FEATURES_PATH = DATA_DIR / "train_values.csv"
    TRAINING_LABELS_PATH = DATA_DIR / "train_labels.csv"
    TEST_FEATURES_PATH = DATA_DIR / "test_values.csv"
    SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features_df         = pd.read_csv(TRAINING_FEATURES_PATH,   index_col=0)
    labels_df           = pd.read_csv(TRAINING_LABELS_PATH,     index_col=0) - 1
    test_features_df    = pd.read_csv(TEST_FEATURES_PATH,       index_col=0)

    geo_level_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    numerical_columns = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    categorical_columns = ['foundation_type', 'ground_floor_type', 'land_surface_condition', 
                        'legal_ownership_status', 'other_floor_type',
                        'plan_configuration', 'position', 'roof_type']

    # Load All Label Encoders
    with open(MODEL_DIR / 'geo-lv-1-label-encoder.pickle', 'rb') as f:
        le1 = pickle.load(f)
    with open(MODEL_DIR / 'geo-lv-2-label-encoder.pickle', 'rb') as f:
        le2 = pickle.load(f)
    with open(MODEL_DIR / 'geo-lv-3-label-encoder.pickle', 'rb') as f:
        le3 = pickle.load(f)

    # Prepare Transformers
    geo_lv1_le = FunctionTransformer(
        func=lambda x: np.array(le1.transform(x.values.ravel())).reshape(-1, 1),
        feature_names_out='one-to-one'
    )

    geo_lv2_le = FunctionTransformer(
        func=lambda x: np.array(le2.transform(x.values.ravel())).reshape(-1, 1), 
        feature_names_out='one-to-one'
    )

    geo_lv3_le = FunctionTransformer(
        func=lambda x: np.array(le3.transform(x.values.ravel())).reshape(-1, 1), 
        feature_names_out='one-to-one'
    )

    # Dim Reducer
    geo_dim_reduction_preprocessor = ColumnTransformer(
        transformers=[
            ('geo1_le', geo_lv1_le, ['geo_level_1_id']),
            ('geo2_le', geo_lv2_le, ['geo_level_2_id']),
            ('geo3_le', geo_lv3_le, ['geo_level_3_id']),
        ], 
        remainder='drop', 
        verbose_feature_names_out=False
    ).set_output(transform='pandas')

    geo_dim_reduction_pipe = Pipeline([
        ('label_encoder', geo_dim_reduction_preprocessor),
        ('embedder', GeoDimensionReduction0(path=MODEL_DIR / 'project-dr-16.pt', device=DEVICE, latent_dim=16)),
    ])


    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore', min_frequency=1, sparse_output=False), categorical_columns + ['geo_level_1_id']),
            ('geo_dim_reduction', geo_dim_reduction_pipe, geo_level_columns),
            ('geo_unk',  ThresholdReplacer(3, -1), geo_level_columns),
        ],
        remainder='passthrough'
    )
    preprocessor.set_output(transform='pandas')

    X_train = preprocessor.fit_transform(features_df, labels_df)
    y_train = labels_df.to_numpy().squeeze()
    X_test = preprocessor.transform(test_features_df)

    X_train_avalible, X_test_avalible, y_train_avalible, y_test_avalible = train_test_split(X_train,y_train,random_state=random_seed, test_size=0.2)

    tuned_parameters = {
        "CatBoost Classifier": {
            'border_count': 13, 'depth': 10, 'iterations': 5000, 'l2_leaf_reg': 3, 'learning_rate': 0.5, 'verbose':0
        },
        "LightGBM Classifier": {
            'colsample_bytree': 0.7, 'max_depth': 20, 'min_split_gain': 0.3, 'n_estimators': 100,
            'num_leaves': 100, 'reg_alpha': 1.2, 'reg_lambda': 1.2, 'subsample': 0.9, 'subsample_freq': 20
        },
        "Gradient Boosting Classifier": {
            'learning_rate': 0.025, 'max_depth': 8, 'max_features': 'log2',
            'subsample': 0.8, 'n_estimators': 400
        },
        "XGBoost Classifier": {
            'booster': 'gbtree', 'colsample_bytree': 0.7000000000000001, 
            'eta': 0.025, 'eval_metric': 'auc', 'gamma': 0.9, 'max_depth': 11, 
            'min_child_weight': 6.0, 'n_estimators': 969, 'nthread': 6, 
            'seed': random_seed, 'subsample': 0.8, 'tree_method': 'gpu_hist'
        },
        "AdaBoost Classifier": {
            'algorithm': 'SAMME.R', 'learning_rate': 0.15, 'n_estimators': 500
        },
        "Random Forest Classifier": {
            'criterion': 'gini', 'max_depth': 8, 'max_features': 'sqrt', 'n_estimators': 500
        },
        "Extra Trees Classifier": {
            'max_depth': 25, 'n_estimators': 500
        },
        "AsymBoost Classifier": {
            'algorithm': 'SAMME.R', 'learning_rate': 1, 'n_estimators': 50
        }
    }

    # Training and evaluating each model with tuned parameters
    tuned_models = {
        "CatBoost Classifier": CatBoostClassifier(**tuned_parameters["CatBoost Classifier"]),
        "LightGBM Classifier": lgb.LGBMClassifier(**tuned_parameters["LightGBM Classifier"]),
        "Gradient Boosting Classifier": GradientBoostingClassifier(**tuned_parameters["Gradient Boosting Classifier"]),
        "XGBoost Classifier": XGBClassifier(**tuned_parameters["XGBoost Classifier"]),
        "AdaBoost Classifier": AdaBoostClassifier(**tuned_parameters["AdaBoost Classifier"]),
        "Random Forest Classifier": RandomForestClassifier(**tuned_parameters["Random Forest Classifier"]),
        "Extra Trees Classifier": ExtraTreesClassifier(**tuned_parameters["Extra Trees Classifier"]),
        "AsymBoost Classifier": AsymBoostClassifier(**tuned_parameters["AsymBoost Classifier"])
    }
    print("\n\n")
    # Training and evaluating each model with tuned parameters
    for name, tuned_model in tqdm(tuned_models.items()):
        tuned_model.fit(X_train_avalible, y_train_avalible)
        y_pred_tuned = tuned_model.predict(X_test_avalible)
        tuned_score = f1_score(y_pred_tuned, y_test_avalible, average='micro')
        print(f"{name} with tuning: {tuned_score}")

train_algorithm(2762)