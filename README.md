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
