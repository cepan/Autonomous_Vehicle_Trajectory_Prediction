# UCSD_CSE151B_Kaggle
UCSD Deep Learning Class Competition Autonomous vehicle motion forecasting challenge

## How to get started with this project

- Step 1: Go to [Kaggle page](https://www.kaggle.com/competitions/cse151b-spring2022/data) and downalod datset into your local/ cloud machine.
- Step 2: Move the `test` and `train` folder into `argo` folder (since I used git ignore to prevent data from being uploaded to git, you need to download and set up data for the first time).
  - The file structure is supposed to look like this.
  - ![sample_file_structure](readme_imgs\sample_file_structure.png)
  - Please follow this structure, or else GitHub will prevent push request because large file size.
- Step 3: Start using this `Load_Argo2_Public.ipynb` to start model training and testing.

## Todos:
1. Preprocessing data
2. Discuss models to try
3. Try submit some results to Kaggle

## Model to try:
1. RNN
2. LSTM
3. Transformer
4. Neural ODE