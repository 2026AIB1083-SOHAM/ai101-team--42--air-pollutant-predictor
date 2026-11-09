# Team-42-Air-Pollutant-Predictor
Predicts air pollutant concentrations (e.g., CO / Benzene) from multi-sensor and meteorological data using linear regression (numpy,pandas, scikit-learn, matplotlib).
### Team and roles

| Member | Entry no. | Role |
| :--- | :--- | :--- |
| Soham Gupta | 2026AIB1083 | Documentation lead |
| Amarpreet Singh | 2026EEB1318 | Execution lead |
| Kedare Ashish Rajendra | 2026MCB1535 | PPT lead |
| Nihal Krishna Chetterbhaktula | 2026DAB1299 | Video lead |

*All four members contributed code (see commit history).*

### Project structure

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

### Setup

Tested on Python 3.10, 3.11, and 3.14 (Windows, macOS, Ubuntu).

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/ai101-team-number-air-pollutant-predictor.git
cd ai101-team-number-air-pollutant-predictor

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows (Command Prompt / PowerShell):
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

### How to run

1. Run the complete pipeline from the project root directory:
   ```bash
   python src/main.py
   ```
2. The script automatically executes data ingestion and cleaning, IQR outlier detection, linear regression model training, metric evaluation (MAE, MSE, RMSE, R²), and exports 7 plots into `outputs/`.
3. Expected runtime: under 35 seconds.