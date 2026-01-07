"""
Generate complex, realistic demo files for Context Thread Agent
Creates both comprehensive Jupyter notebooks and Excel files
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


def create_complex_sales_analysis_excel():
    """Create a complex multi-sheet Excel workbook with sales analysis data"""
    
    np.random.seed(42)
    
    # Sheet 1: Raw Sales Data (500 rows)
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D', 'Product_E', 
                'Product_F', 'Product_G', 'Product_H']
    
    raw_data = []
    for _ in range(500):
        raw_data.append({
            'Date': np.random.choice(dates),
            'Region': np.random.choice(regions),
            'Product': np.random.choice(products),
            'Units_Sold': np.random.randint(10, 500),
            'Unit_Price': np.round(np.random.uniform(50, 500), 2),
            'Cost_Per_Unit': np.round(np.random.uniform(20, 250), 2),
            'Discount_Pct': np.round(np.random.uniform(0, 0.25), 2),
            'Sales_Rep_ID': f"REP_{np.random.randint(1, 50):03d}",
            'Customer_Segment': np.random.choice(['Enterprise', 'SMB', 'Consumer']),
            'Payment_Method': np.random.choice(['Credit', 'Cash', 'Wire', 'Check'])
        })
    
    df_raw = pd.DataFrame(raw_data)
    df_raw['Revenue'] = df_raw['Units_Sold'] * df_raw['Unit_Price'] * (1 - df_raw['Discount_Pct'])
    df_raw['Profit'] = df_raw['Revenue'] - (df_raw['Units_Sold'] * df_raw['Cost_Per_Unit'])
    df_raw['Profit_Margin'] = (df_raw['Profit'] / df_raw['Revenue'] * 100).round(2)
    
    # Sheet 2: Regional Summary
    regional_summary = df_raw.groupby('Region').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Units_Sold': 'sum',
        'Date': 'count'
    }).round(2)
    regional_summary.columns = ['Total_Revenue', 'Total_Profit', 'Total_Units', 'Transaction_Count']
    regional_summary['Avg_Transaction_Value'] = (regional_summary['Total_Revenue'] / 
                                                   regional_summary['Transaction_Count']).round(2)
    
    # Sheet 3: Product Performance
    product_perf = df_raw.groupby('Product').agg({
        'Revenue': ['sum', 'mean'],
        'Profit': ['sum', 'mean'],
        'Units_Sold': 'sum',
        'Profit_Margin': 'mean'
    }).round(2)
    product_perf.columns = ['_'.join(col) for col in product_perf.columns]
    
    # Sheet 4: Time Series Analysis
    df_raw['YearMonth'] = pd.to_datetime(df_raw['Date']).dt.to_period('M')
    time_series = df_raw.groupby('YearMonth').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Units_Sold': 'sum'
    }).round(2)
    time_series.index = time_series.index.astype(str)
    
    # Sheet 5: Sales Rep Performance
    rep_perf = df_raw.groupby('Sales_Rep_ID').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Date': 'count',
        'Profit_Margin': 'mean'
    }).round(2)
    rep_perf.columns = ['Total_Revenue', 'Total_Profit', 'Transactions', 'Avg_Margin']
    rep_perf = rep_perf.sort_values('Total_Revenue', ascending=False).head(30)
    
    # Sheet 6: Anomaly Detection
    anomalies = df_raw[
        (df_raw['Profit_Margin'] < 0) | 
        (df_raw['Profit_Margin'] > 80) |
        (df_raw['Discount_Pct'] > 0.20)
    ].copy()
    anomalies['Flag'] = anomalies.apply(lambda x: 
        'Negative Margin' if x['Profit_Margin'] < 0 else
        'High Margin' if x['Profit_Margin'] > 80 else
        'High Discount', axis=1)
    
    # Write to Excel
    output_path = Path('demo_files/complex_sales_analysis.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_raw.to_excel(writer, sheet_name='Raw_Sales_Data', index=False)
        regional_summary.to_excel(writer, sheet_name='Regional_Summary')
        product_perf.to_excel(writer, sheet_name='Product_Performance')
        time_series.to_excel(writer, sheet_name='Monthly_Trends')
        rep_perf.to_excel(writer, sheet_name='Top_Sales_Reps')
        anomalies.to_excel(writer, sheet_name='Anomalies', index=False)
    
    print(f"✅ Created complex sales analysis Excel: {output_path}")
    return output_path


def create_financial_modeling_excel():
    """Create a financial modeling workbook with multiple interconnected sheets"""
    
    np.random.seed(123)
    
    # Sheet 1: Income Statement
    years = ['2021', '2022', '2023', '2024F', '2025F']
    income_stmt = pd.DataFrame({
        'Line_Item': [
            'Revenue', 'Cost_of_Goods_Sold', 'Gross_Profit', 'R&D_Expense',
            'Sales_Marketing', 'General_Admin', 'Operating_Income',
            'Interest_Expense', 'Tax_Expense', 'Net_Income'
        ]
    })
    
    base_revenue = 50000000
    for i, year in enumerate(years):
        growth = 1.15 if 'F' in year else 1.12
        revenue = base_revenue * (growth ** i)
        cogs = revenue * 0.35
        gross_profit = revenue - cogs
        rd = revenue * 0.15
        sm = revenue * 0.25
        ga = revenue * 0.12
        op_income = gross_profit - rd - sm - ga
        interest = 2000000
        tax = op_income * 0.21 if op_income > 0 else 0
        net_income = op_income - interest - tax
        
        income_stmt[year] = [
            revenue, cogs, gross_profit, rd, sm, ga,
            op_income, interest, tax, net_income
        ]
    
    # Sheet 2: Balance Sheet
    balance_sheet = pd.DataFrame({
        'Line_Item': [
            'Cash', 'Accounts_Receivable', 'Inventory', 'Total_Current_Assets',
            'PP&E_Net', 'Intangibles', 'Total_Assets',
            'Accounts_Payable', 'Short_Term_Debt', 'Total_Current_Liabilities',
            'Long_Term_Debt', 'Total_Liabilities', 'Shareholders_Equity'
        ]
    })
    
    for i, year in enumerate(years):
        cash = 10000000 * (1.1 ** i)
        ar = 8000000 * (1.12 ** i)
        inventory = 5000000 * (1.08 ** i)
        current_assets = cash + ar + inventory
        ppe = 30000000 * (1.05 ** i)
        intangibles = 15000000
        total_assets = current_assets + ppe + intangibles
        
        ap = 6000000 * (1.1 ** i)
        std = 5000000
        current_liab = ap + std
        ltd = 25000000
        total_liab = current_liab + ltd
        equity = total_assets - total_liab
        
        balance_sheet[year] = [
            cash, ar, inventory, current_assets, ppe, intangibles, total_assets,
            ap, std, current_liab, ltd, total_liab, equity
        ]
    
    # Sheet 3: Cash Flow Statement
    cashflow = pd.DataFrame({
        'Line_Item': [
            'Net_Income', 'Depreciation', 'Changes_in_Working_Capital',
            'Operating_Cash_Flow', 'Capital_Expenditures', 'Investing_Cash_Flow',
            'Debt_Issuance', 'Dividends_Paid', 'Financing_Cash_Flow',
            'Net_Change_in_Cash'
        ]
    })
    
    for i, year in enumerate(years):
        net_income = income_stmt[year][9]
        depreciation = 3000000
        wc_change = -2000000 * (1.1 ** i)
        ocf = net_income + depreciation + wc_change
        capex = -5000000
        icf = capex
        debt_iss = 0 if i < 3 else 10000000
        dividends = -1500000
        fcf = debt_iss + dividends
        net_cash = ocf + icf + fcf
        
        cashflow[year] = [
            net_income, depreciation, wc_change, ocf,
            capex, icf, debt_iss, dividends, fcf, net_cash
        ]
    
    # Sheet 4: Key Ratios
    ratios = pd.DataFrame({
        'Ratio': [
            'Revenue_Growth_%', 'Gross_Margin_%', 'Operating_Margin_%',
            'Net_Margin_%', 'ROE_%', 'Current_Ratio', 'Debt_to_Equity',
            'Interest_Coverage'
        ]
    })
    
    for year in years:
        revenue = income_stmt[year][0]
        gross = income_stmt[year][2]
        op_income = income_stmt[year][6]
        net = income_stmt[year][9]
        equity = balance_sheet[year][12]
        current_assets = balance_sheet[year][3]
        current_liab = balance_sheet[year][9]
        total_debt = balance_sheet[year][11]
        interest = income_stmt[year][7]
        
        ratios[year] = [
            0 if year == '2021' else 15.0,
            (gross / revenue * 100) if revenue > 0 else 0,
            (op_income / revenue * 100) if revenue > 0 else 0,
            (net / revenue * 100) if revenue > 0 else 0,
            (net / equity * 100) if equity > 0 else 0,
            current_assets / current_liab if current_liab > 0 else 0,
            total_debt / equity if equity > 0 else 0,
            op_income / interest if interest > 0 else 0
        ]
    
    output_path = Path('demo_files/financial_model.xlsx')
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        income_stmt.to_excel(writer, sheet_name='Income_Statement', index=False)
        balance_sheet.to_excel(writer, sheet_name='Balance_Sheet', index=False)
        cashflow.to_excel(writer, sheet_name='Cash_Flow', index=False)
        ratios.to_excel(writer, sheet_name='Key_Ratios', index=False)
    
    print(f"✅ Created financial modeling Excel: {output_path}")
    return output_path


def create_complex_ml_notebook():
    """Create a comprehensive ML analysis notebook"""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    cells = [
        # Cell 1: Title
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Advanced Customer Churn Prediction Analysis\n",
                      "\n",
                      "## Business Context\n",
                      "This notebook analyzes customer churn patterns for a telecommunications company.\n",
                      "Key objectives:\n",
                      "- Identify high-risk customers\n",
                      "- Understand churn drivers\n",
                      "- Build predictive models\n",
                      "- Recommend retention strategies\n",
                      "\n",
                      "**Dataset:** 10,000 customers with 50+ features\n",
                      "**Target:** Binary churn indicator (Yes/No)"]
        },
        # Cell 2: Imports
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV\n",
                "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
                "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve\n",
                "from sklearn.feature_selection import SelectKBest, f_classif\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Set display options\n",
                "pd.set_option('display.max_columns', None)\n",
                "sns.set_style('whitegrid')\n",
                "plt.rcParams['figure.figsize'] = (12, 6)\n",
                "\n",
                "print('Libraries imported successfully')\n",
                "print(f'Pandas version: {pd.__version__}')\n",
                "print(f'NumPy version: {np.__version__}')"
            ]
        },
        # Cell 3: Load Data
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Dataset shape: (10000, 52)\n",
                        "Churn rate: 26.5%\n"
                    ]
                }
            ],
            "source": [
                "# Generate synthetic customer data\n",
                "np.random.seed(42)\n",
                "n_customers = 10000\n",
                "\n",
                "data = {\n",
                "    'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],\n",
                "    'tenure_months': np.random.randint(1, 72, n_customers),\n",
                "    'monthly_charges': np.random.uniform(20, 150, n_customers),\n",
                "    'total_charges': np.random.uniform(100, 8000, n_customers),\n",
                "    'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], n_customers),\n",
                "    'payment_method': np.random.choice(['Electronic', 'Mailed Check', 'Bank Transfer', 'Credit Card'], n_customers),\n",
                "    'internet_service': np.random.choice(['DSL', 'Fiber Optic', 'No'], n_customers),\n",
                "    'online_security': np.random.choice(['Yes', 'No', 'No internet'], n_customers),\n",
                "    'tech_support': np.random.choice(['Yes', 'No', 'No internet'], n_customers),\n",
                "    'streaming_tv': np.random.choice(['Yes', 'No', 'No internet'], n_customers),\n",
                "    'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),\n",
                "    'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.84, 0.16]),\n",
                "    'partner': np.random.choice(['Yes', 'No'], n_customers),\n",
                "    'dependents': np.random.choice(['Yes', 'No'], n_customers),\n",
                "    'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),\n",
                "    'multiple_lines': np.random.choice(['Yes', 'No', 'No phone'], n_customers),\n",
                "}\n",
                "\n",
                "df = pd.DataFrame(data)\n",
                "\n",
                "# Create churn with logical patterns\n",
                "churn_probability = 0.1  # Base probability\n",
                "churn_probability += (df['tenure_months'] < 12) * 0.3  # New customers more likely\n",
                "churn_probability += (df['contract_type'] == 'Month-to-Month') * 0.25\n",
                "churn_probability += (df['monthly_charges'] > 100) * 0.15\n",
                "churn_probability += (df['tech_support'] == 'No') * 0.1\n",
                "churn_probability = np.clip(churn_probability, 0, 1)\n",
                "\n",
                "df['churn'] = np.random.binomial(1, churn_probability)\n",
                "\n",
                "print(f'Dataset shape: {df.shape}')\n",
                "print(f'Churn rate: {df.churn.mean()*100:.1f}%')"
            ]
        },
        # Cell 4: EDA
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Exploratory Data Analysis\n",
                      "\n",
                      "### Key Questions:\n",
                      "1. What is the churn rate?\n",
                      "2. Which features correlate with churn?\n",
                      "3. Are there any data quality issues?\n",
                      "4. What patterns exist in churned vs retained customers?"]
        },
        # Cell 5: Data Quality Check
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Missing values:\n",
                        "customer_id         0\n",
                        "tenure_months       0\n",
                        "monthly_charges     0\n",
                        "total_charges       0\n",
                        "churn               0\n",
                        "dtype: int64\n"
                    ]
                }
            ],
            "source": [
                "# Check for missing values\n",
                "print('Missing values:')\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Check for duplicates\n",
                "print(f'\\nDuplicate rows: {df.duplicated().sum()}')\n",
                "\n",
                "# Data types\n",
                "print('\\nData types:')\n",
                "print(df.dtypes)\n",
                "\n",
                "# Basic statistics\n",
                "print('\\nNumerical features summary:')\n",
                "print(df.describe())"
            ]
        },
        # Cell 6: Feature Engineering
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Create additional features\n",
                "df['avg_monthly_charges'] = df['total_charges'] / df['tenure_months'].replace(0, 1)\n",
                "df['tenure_group'] = pd.cut(df['tenure_months'], bins=[0, 12, 24, 48, 72], \n",
                "                            labels=['0-1 year', '1-2 years', '2-4 years', '4+ years'])\n",
                "df['charge_per_tenure'] = df['total_charges'] / (df['tenure_months'] + 1)\n",
                "df['is_new_customer'] = (df['tenure_months'] <= 6).astype(int)\n",
                "df['high_charges'] = (df['monthly_charges'] > df['monthly_charges'].median()).astype(int)\n",
                "\n",
                "# Encode categorical variables\n",
                "label_encoders = {}\n",
                "categorical_cols = df.select_dtypes(include=['object']).columns.tolist()\n",
                "categorical_cols.remove('customer_id')\n",
                "\n",
                "for col in categorical_cols:\n",
                "    if col != 'tenure_group':\n",
                "        le = LabelEncoder()\n",
                "        df[f'{col}_encoded'] = le.fit_transform(df[col])\n",
                "        label_encoders[col] = le\n",
                "\n",
                "print('Feature engineering completed')\n",
                "print(f'Total features: {df.shape[1]}')"
            ]
        },
        # Cell 7: Model Building
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Training set size: 7000\n",
                        "Test set size: 3000\n",
                        "\\nRandom Forest Accuracy: 0.847\n",
                        "Random Forest AUC: 0.891\n"
                    ]
                }
            ],
            "source": [
                "# Prepare features for modeling\n",
                "feature_cols = [col for col in df.columns if col.endswith('_encoded') or \n",
                "                df[col].dtype in ['int64', 'float64']]\n",
                "feature_cols = [col for col in feature_cols if col not in ['customer_id', 'churn']]\n",
                "\n",
                "X = df[feature_cols]\n",
                "y = df['churn']\n",
                "\n",
                "# Train-test split\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \n",
                "                                                    random_state=42, stratify=y)\n",
                "\n",
                "print(f'Training set size: {len(X_train)}')\n",
                "print(f'Test set size: {len(X_test)}')\n",
                "\n",
                "# Scale features\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = scaler.fit_transform(X_train)\n",
                "X_test_scaled = scaler.transform(X_test)\n",
                "\n",
                "# Train Random Forest\n",
                "rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, \n",
                "                                 random_state=42, n_jobs=-1)\n",
                "rf_model.fit(X_train_scaled, y_train)\n",
                "\n",
                "# Evaluate\n",
                "y_pred = rf_model.predict(X_test_scaled)\n",
                "y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]\n",
                "\n",
                "accuracy = rf_model.score(X_test_scaled, y_test)\n",
                "auc = roc_auc_score(y_test, y_pred_proba)\n",
                "\n",
                "print(f'\\nRandom Forest Accuracy: {accuracy:.3f}')\n",
                "print(f'Random Forest AUC: {auc:.3f}')"
            ]
        },
        # Cell 8: Feature Importance
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Get feature importance\n",
                "feature_importance = pd.DataFrame({\n",
                "    'feature': feature_cols,\n",
                "    'importance': rf_model.feature_importances_\n",
                "}).sort_values('importance', ascending=False)\n",
                "\n",
                "print('Top 10 Most Important Features:')\n",
                "print(feature_importance.head(10))\n",
                "\n",
                "# Plot feature importance\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.barh(feature_importance.head(15)['feature'], \n",
                "         feature_importance.head(15)['importance'])\n",
                "plt.xlabel('Importance')\n",
                "plt.title('Top 15 Feature Importances')\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        # Cell 9: Conclusions
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Key Findings\n",
                "\n",
                "### Model Performance\n",
                "- **Accuracy**: 84.7%\n",
                "- **AUC-ROC**: 0.891\n",
                "- The model shows strong predictive power\n",
                "\n",
                "### Churn Drivers\n",
                "1. **Contract Type**: Month-to-month contracts have highest churn\n",
                "2. **Tenure**: New customers (< 12 months) are at highest risk\n",
                "3. **Charges**: High monthly charges correlate with churn\n",
                "4. **Services**: Lack of tech support increases churn probability\n",
                "\n",
                "### Business Recommendations\n",
                "1. **Focus on new customer onboarding** (first 6-12 months)\n",
                "2. **Incentivize longer contracts** (annual vs monthly)\n",
                "3. **Bundle tech support** with high-value packages\n",
                "4. **Monitor customers with monthly charges > $100**\n",
                "5. **Implement early warning system** using this model\n",
                "\n",
                "### Next Steps\n",
                "- A/B test retention campaigns\n",
                "- Deploy model to production\n",
                "- Monitor model performance monthly\n",
                "- Collect additional behavioral data"
            ]
        }
    ]
    
    notebook["cells"] = cells
    
    output_path = Path('demo_files/customer_churn_analysis.ipynb')
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Created complex ML notebook: {output_path}")
    return output_path


def create_time_series_notebook():
    """Create a time series forecasting notebook"""
    
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Stock Price Forecasting with ARIMA and LSTM\n",
                "\n",
                "## Objective\n",
                "Build and compare time series forecasting models for stock price prediction.\n",
                "\n",
                "**Dataset**: Daily stock prices (5 years)\n",
                "**Models**: ARIMA, SARIMA, LSTM\n",
                "**Metrics**: RMSE, MAE, MAPE"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from statsmodels.tsa.arima.model import ARIMA\n",
                "from statsmodels.tsa.stattools import adfuller, acf, pacf\n",
                "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n",
                "from sklearn.metrics import mean_squared_error, mean_absolute_error\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "\n",
                "# Generate synthetic stock data\n",
                "np.random.seed(42)\n",
                "dates = pd.date_range('2019-01-01', '2024-01-01', freq='D')\n",
                "n = len(dates)\n",
                "\n",
                "# Generate realistic stock price with trend, seasonality, and noise\n",
                "trend = np.linspace(100, 200, n)\n",
                "seasonal = 10 * np.sin(np.linspace(0, 10*np.pi, n))\n",
                "noise = np.random.normal(0, 5, n)\n",
                "prices = trend + seasonal + noise\n",
                "prices = np.maximum(prices, 50)  # Ensure positive prices\n",
                "\n",
                "df = pd.DataFrame({\n",
                "    'Date': dates,\n",
                "    'Close': prices,\n",
                "    'Volume': np.random.randint(1000000, 10000000, n)\n",
                "})\n",
                "df.set_index('Date', inplace=True)\n",
                "\n",
                "print(f'Dataset shape: {df.shape}')\n",
                "print(f'Date range: {df.index.min()} to {df.index.max()}')\n",
                "print(f'Mean price: ${df.Close.mean():.2f}')\n",
                "print(f'Price volatility (std): ${df.Close.std():.2f}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Stationarity test\n",
                "result = adfuller(df['Close'])\n",
                "print('ADF Statistic:', result[0])\n",
                "print('p-value:', result[1])\n",
                "print('Critical Values:', result[4])\n",
                "\n",
                "if result[1] > 0.05:\n",
                "    print('\\nSeries is NON-STATIONARY. Differencing required.')\n",
                "    df['Close_diff'] = df['Close'].diff().dropna()\n",
                "else:\n",
                "    print('\\nSeries is STATIONARY.')\n",
                "\n",
                "# Calculate returns\n",
                "df['Returns'] = df['Close'].pct_change() * 100\n",
                "df['MA_7'] = df['Close'].rolling(window=7).mean()\n",
                "df['MA_30'] = df['Close'].rolling(window=30).mean()\n",
                "\n",
                "print(f'\\nAverage daily return: {df.Returns.mean():.3f}%')\n",
                "print(f'Return volatility: {df.Returns.std():.3f}%')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Train-test split (80-20)\n",
                "train_size = int(len(df) * 0.8)\n",
                "train, test = df[:train_size], df[train_size:]\n",
                "\n",
                "print(f'Training set: {len(train)} days')\n",
                "print(f'Test set: {len(test)} days')\n",
                "\n",
                "# Fit ARIMA model\n",
                "model = ARIMA(train['Close'], order=(5,1,2))\n",
                "model_fit = model.fit()\n",
                "\n",
                "print('\\nARIMA Model Summary:')\n",
                "print(model_fit.summary())\n",
                "\n",
                "# Forecast\n",
                "forecast = model_fit.forecast(steps=len(test))\n",
                "test['Forecast'] = forecast.values\n",
                "\n",
                "# Calculate errors\n",
                "rmse = np.sqrt(mean_squared_error(test['Close'], test['Forecast']))\n",
                "mae = mean_absolute_error(test['Close'], test['Forecast'])\n",
                "mape = np.mean(np.abs((test['Close'] - test['Forecast']) / test['Close'])) * 100\n",
                "\n",
                "print(f'\\nModel Performance:')\n",
                "print(f'RMSE: ${rmse:.2f}')\n",
                "print(f'MAE: ${mae:.2f}')\n",
                "print(f'MAPE: {mape:.2f}%')"
            ]
        }
    ]
    
    notebook["cells"] = cells
    
    output_path = Path('demo_files/stock_forecasting.ipynb')
    with open(output_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"✅ Created time series notebook: {output_path}")
    return output_path


if __name__ == "__main__":
    print("🎯 Generating Complex Demo Files...\n")
    
    # Generate Excel files
    create_complex_sales_analysis_excel()
    create_financial_modeling_excel()
    
    # Generate Jupyter notebooks
    create_complex_ml_notebook()
    create_time_series_notebook()
    
    print("\n✅ All demo files created successfully!")
    print("\nGenerated files:")
    print("- demo_files/complex_sales_analysis.xlsx (6 sheets, 500+ rows)")
    print("- demo_files/financial_model.xlsx (4 sheets, financial statements)")
    print("- demo_files/customer_churn_analysis.ipynb (ML analysis, 200+ lines)")
    print("- demo_files/stock_forecasting.ipynb (Time series analysis)")
