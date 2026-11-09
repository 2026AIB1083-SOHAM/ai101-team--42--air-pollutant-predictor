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

## Project structure
ai101-team-<number>-air-pollutant-predictor/
├── data/
│   └── air_quality.csv          # 9,357 rows, UCI multi-sensor & meteorological readings[cite: 1, 5]
├── outputs/                     # 7 generated evaluation & EDA plots saved here
├── src/
│   ├── cleaning.py              # Ingestion, formatting & -200 sentinel missing value cleanup[cite: 1, 5]
│   ├── visualization.py         # 1.5*IQR outlier boxplot & correlation bar chart[cite: 4, 5]
│   ├── regression_model.py      # Parametric OLS Linear Regression training[cite: 1, 5, 8]
│   ├── model_evaluation.py     # Metric logging (MAE, RMSE, R²) & residual diagnostics[cite: 5, 8]
│   └── main.py                  # Entry point orchestrating full pipeline[cite: 5]
├── .gitignore                   # Excludes venv/, __pycache__/, and OS files[cite: 5]
├── requirements.txt             # Pinned package versions[cite: 5]
└── README.md                    # Project documentation & reproducibility guide[cite: 5]
