import os
from typing import List
import pandas as pd
from tqdm.notebook import tqdm

def read_dataset(dataset_path: str) -> List[pd.DataFrame]:  
    ret = []
    for name in tqdm(os.listdir(dataset_path)):
        path = os.path.join(dataset_path, name)
        ret.append(pd.read_parquet(path, engine='pyarrow'))
    return ret
