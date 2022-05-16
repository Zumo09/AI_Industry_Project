import os
from typing import List
import pandas as pd
from tqdm.notebook import tqdm

DATASET_PATH = os.path.join(os.getcwd(), "new_data")

def read_dataset() -> List[pd.DataFrame]:  
    ret = []
    for name in tqdm(os.listdir(DATASET_PATH)):
        path = os.path.join(DATASET_PATH, name)
        ret.append(pd.read_parquet(path, engine='pyarrow'))
    return ret
