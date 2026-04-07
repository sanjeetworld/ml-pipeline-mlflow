import pandas as pd
import numpy as np

np.random.seed(42)

rows = 10000  # large dataset

data = {
    "age": np.random.randint(18, 60, rows),
    "salary": np.random.randint(20000, 150000, rows),
    "experience": np.random.randint(0, 20, rows),
    "education_level": np.random.choice([0,1,2], rows),  # 0=UG,1=PG,2=PhD
    "city_tier": np.random.choice([1,2,3], rows),
}

df = pd.DataFrame(data)

# Target logic (realistic)
df["purchased"] = (
    (df["salary"] > 60000) &
    (df["experience"] > 2)
).astype(int)

df.to_csv("data/data.csv", index=False)

print("Dataset created!")