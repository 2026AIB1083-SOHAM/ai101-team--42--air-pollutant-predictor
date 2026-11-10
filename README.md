# Air Pollutant Predictor using Linear Regression

An educational machine-learning project that predicts air pollutant concentrations (e.g., CO / Benzene) from multi-sensor and meteorological data using linear regression (numpy,pandas, scikit-learn, matplotlib).
>**Important:** This is an academic demonstration, not an official >predictor tool or decision support tool
### Team and roles

| Member |IIT ROPAR Entry no. | Role |
| :--- | :--- | :--- |
| Soham Gupta | 2026AIB1083 | Documentation lead |
| Amarpreet Singh | 2026EEB1318 | Execution lead |
| Kedare Ashish Rajendra | 2026MCB1535 | PPT lead |
| Nihal Krishna Chetterbhaktula | 2026DAB1299 | Video lead |

*All four members contributed code (see commit history).*
**GitHub repository:**
<put the updated>

## Project Overview

The project uses the UCI air quality dataset to identify trends and develop a Linear regression model to predict CO(GT). 

The model uses 7 input features.

## Dataset 

The supplied CSV file is 'air_quality.csv' 
*See the data folder* 
- **Rows:** 9,357 hourly records
- **Columns:** 15
- **Target:** Pollutant concentration
- **Missing values:** We removed empty columns and replaced the dataset's −200 "missing" code with blanks.

Do not rename files unless you also
update the path in the commands accordingly.

## Project structure

```text
ai101-team-number-air-pollutant-predictor/
├── data/
│   └── air_quality.csv          # 9,357 rows, UCI multi-sensor & meteorological data
├── outputs/                     # 7 generated evaluation plots saved here
├── src/
│   ├── cleaning.py              # data ingestion and cleaning
│   ├── visualization.py         # 1.5*IQR outlier boxplot & correlation analysis
│   ├── regression_model.py      # Parametric OLS Linear Regression training
│   ├── model_evaluation.py      # Evaluation metrics (MAE, RMSE, R²) & residual plots
│   └── main.py                  # Entry point orchestrating full pipeline
├── .gitignore                   
├── requirements.txt             # Pinned package dependencies
└── README.md                    # Documentation & reproducibility instructions
```
## Setup

Tested on Python 3.10, 3.11, and 3.14 (Windows, Ubuntu).

#### 1. Clone the repository
"Please run the code one line at a time"

```bash
git clone https://github.com/<your-username>/ai101-team-number-air-pollutant-predictor.git
cd ai101-team-number-air-pollutant-predicto
```


#### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

### 2. Create and activate a virtual environment

**Windows (Command Prompt):**

``` bat
py -m venv .venv
.venv\Scripts\activate
```

**Windows (PowerShell):**

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install required dependencies

### 3. Install dependencies

``` bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

### How to run

1. Run the complete pipeline from the project root directory:

   ```bash
   python src/main.py
   ```

2. The script automatically executes data ingestion and cleaning, IQR outlier detection, linear regression model training, metric evaluation (MAE, MSE, RMSE, R²), and exports 7 plots into `outputs/`.

3. Expected runtime: under 35 seconds.

## Model Results also include screen shot here
## Model Results

The model was executed with the supplied dataset (`data/air_quality.csv`) and the dependency versions listed in `requirements.txt`[cite: 3, 5]. With an 80/20 train/test split (random seed `42`) and median binarization threshold, the testing results were[cite: 4, 5]:

| Metric | Test result |
| :--- | ---: |
| MAE | 0.384 |
| RMSE | 0.530 |
| R² | 0.885 |
| Accuracy | 0.852 |
| Precision | 0.861 |
| Recall | 0.840 |
| F1 score | 0.850 |

The test confusion matrix for binary classification (0 = Safe, 1 = Polluted) is

```text
[[782  126]
 [145  740]]

## Presentation and Video

- **Project presentation (PPT):** `[shareable PPT link here]`
- **Project video (maximum 6 minutes):**
  `[shareable video link here]`


## AI-Use Disclosure

Gemini (Google) was used to assist with drafting and organising this
README and reviewing the project files for documentation details, and was also used to research for the os and sys python libraries that make the code more robust . The
project description, repository instructions, and reported metrics were
checked against the supplied code files and dataset; the code files were also
executed to verify that it runs with the pinned dependencies listed in
`requirements.txt`. Additionally gemini was used to refine the comments in the code files.

Claude (Anthropic) was used to refine the model training and evaluation process.
