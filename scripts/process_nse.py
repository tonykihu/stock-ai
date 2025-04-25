import pandas as pd
import glob
import os

latest_file = max(glob.glob("data/kenya/nse_2020.csv"), key=os.path.getctime)
df = pd.read_csv(latest_file)
df.to_csv("data/kenya/nse_processed.csv", index=False)
print(f"Latest file: {latest_file}")