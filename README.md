# Databento lvl3 orderbook feature extration with symbol filter and interactive correlation matrix
# Order Book Feature Analysis

Analyze correlations between extracted features from Databento Level 3 order book data. 

HTML interactive matrix was created for exploring big datasets and to improve color visibility.

## Features Extracted

### Basic Price Statistics
For each symbol, action, and side combination:
- `mean_price_{symbol}_{action}_{side}`
- `median_price_{symbol}_{action}_{side}`
- `max_price_{symbol}_{action}_{side}`
- `min_price_{symbol}_{action}_{side}`
- `var_price_{symbol}_{action}_{side}`
- `std_price_{symbol}_{action}_{side}`
- `open_price_{symbol}_{action}_{side}`
- `close_price_{symbol}_{action}_{side}`
- `price_change_{symbol}_{action}_{side}`
- `unique_price_levels_{symbol}_{action}_{side}`

### Advanced Features
- 22 Catch22 time series features
- Data aggregation with 'right'/'left' labeling for predefined timeframe

## Output
- Static correlation heatmap (PNG)
- Interactive correlation explorer (HTML)
- Filtered feature dataset (CSV)
- Correlation matrix (CSV)

## Future Improvements
- Real-time processing
- Adding new features (potentially - https://github.com/benfulcher/hctsa but in python version)
- GPU speed up

## Example Output

![Correlation Matrix](6EH5_output/correlation_matrix_6EH5_static.png)

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.

## Usage

```bash
python databento_feature_interactive_html_corr_matrix.py
