"""
Order Book Feature Correlation Analysis
Author: [sklep fuzja]
Date: [2025.11.15]

This script analyzes correlations between extracted order book features from Databento L3 data.
It creates both static and interactive correlation matrix visualizations.
"""

import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio

# Configuration
SYMBOL_FILTER = '6EH5'  # Change this to filter for different instruments
OUTPUT_DIR = f'{SYMBOL_FILTER}_output'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created: {OUTPUT_DIR}")

# Load data
df = pd.read_csv('databentofeature_30T_1_test_1.csv', low_memory=False)

print("First rows of data before filtering:")
print(df.head())
print(f"\nNumber of columns before filtering: {len(df.columns)}")
print(f"Columns before filtering: {list(df.columns)}")

# FILTER COLUMNS CONTAINING SPECIFIED SYMBOL IN NAME
columns_with_symbol = [col for col in df.columns if SYMBOL_FILTER in col]
print(f"\nColumns containing '{SYMBOL_FILTER}': {columns_with_symbol}")
print(f"Number of '{SYMBOL_FILTER}' columns: {len(columns_with_symbol)}")

# Create new DataFrame with only columns containing the specified symbol
df_filtered = df[columns_with_symbol].copy()

print(f"\nFiltered DataFrame shape: {df_filtered.shape}")
print("First rows after filtering:")
print(df_filtered.head())

# Check if any columns remain after filtering
if len(columns_with_symbol) == 0:
    print(f"\n⚠️  NO COLUMNS WITH '{SYMBOL_FILTER}' IN NAME!")
    print("Please check if the instrument name is correct.")
else:
    # Continue processing only if there are columns with the specified symbol
    print("\n" + "="*60)
    print(f"PROCESSING DATA WITH {SYMBOL_FILTER} COLUMNS")
    print("="*60)
    
    # Convert 'ts_recv' column if it exists in original df
    if 'ts_recv' in df.columns:
        df_filtered.index = pd.to_datetime(df['ts_recv'])
        # Remove timezone
        df_filtered.index = df_filtered.index.tz_localize(None)
        print("Index set based on 'ts_recv'")
    else:
        print("'ts_recv' column does not exist - skipping index setting")

    print(f"\nData after filtering and setting index:")
    print(df_filtered.head())

    # Check data types
    print("\nData types in filtered columns:")
    print(df_filtered.dtypes)

    # Select only numeric columns
    numeric_columns = df_filtered.select_dtypes(include=['number']).columns
    print(f"\nNumeric columns after filtering: {len(numeric_columns)}")
    print(f"Numeric columns: {list(numeric_columns)}")

    if len(numeric_columns) == 0:
        print("No numeric columns in filtered data!")
    else:
        # Calculate correlation matrix only for numeric columns
        corr_matrix = df_filtered[numeric_columns].corr()

        print(f"\nCorrelation matrix size: {corr_matrix.shape}")
        
        # CALCULATE MEAN CORRELATION
        corr_values = corr_matrix.values
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        
        mean_correlation = np.mean(upper_triangle)
        mean_abs_correlation = np.mean(np.abs(upper_triangle))
        
        print(f"\nMEAN CORRELATION: {mean_correlation:.4f}")
        print(f"MEAN |CORRELATION|: {mean_abs_correlation:.4f}")
        print(f"Correlation range: [{np.min(upper_triangle):.4f}, {np.max(upper_triangle):.4f}]")
        print(f"Number of analyzed pairs: {len(upper_triangle)}")

        # SAVE CORRELATION MATRIX TO CSV
        print("\n" + "="*50)
        print("SAVING CORRELATION MATRIX")
        print("="*50)
        
        filename_corr_csv = os.path.join(OUTPUT_DIR, f'correlation_matrix_{SYMBOL_FILTER}.csv')
        corr_matrix.to_csv(filename_corr_csv)
        print(f"Correlation matrix saved as: {filename_corr_csv}")

        # STATIC VISUALIZATION (MATPLOTLIB/SEABORN)
        print("\n" + "="*50)
        print("CREATING STATIC VISUALIZATION")
        print("="*50)
        
        plt.style.use('default')
        plt.figure(figsize=(12, 10))

        # Create heatmap
        sb.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   linewidths=0.5,
                   linecolor='white',
                   square=True)

        plt.title(f'{SYMBOL_FILTER} Features Correlation Matrix', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save static plot to PNG file
        filename_static = os.path.join(OUTPUT_DIR, f'correlation_matrix_{SYMBOL_FILTER}_static.png')
        plt.savefig(filename_static, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Static plot saved as: {filename_static}")
        
        plt.show()

        # INTERACTIVE VISUALIZATION (PLOTLY)
        print("\n" + "="*50)
        print("CREATING INTERACTIVE VISUALIZATION")
        print("="*50)
        
        # Interactive heatmap
        fig = px.imshow(corr_matrix, 
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title=f'{SYMBOL_FILTER} Features Correlation Matrix (Mean: {mean_correlation:.3f})',
                        labels=dict(color="Correlation"))

        # Adjust layout
        fig.update_layout(
            width=1200, 
            height=1000,
            xaxis_title=f"{SYMBOL_FILTER} Features",
            yaxis_title=f"{SYMBOL_FILTER} Features",
            font=dict(size=10)
        )
        
        # Adjust hover info
        fig.update_traces(
            hovertemplate="<br>".join([
                "Feature X: %{x}",
                "Feature Y: %{y}", 
                "Correlation: %{z:.3f}",
                "<extra></extra>"
            ])
        )
        
        # Save interactive plot
        filename_interactive = os.path.join(OUTPUT_DIR, f'correlation_matrix_{SYMBOL_FILTER}_interactive.html')
        pio.write_html(fig, filename_interactive)
        print(f"Interactive plot saved as: {filename_interactive}")
        
        # Show interactive plot
        try:
            fig.show()
        except Exception as e:
            print(f"Warning: Could not display interactive plot: {e}")

        # Save filtered data to new CSV file
        filename_filtered_csv = os.path.join(OUTPUT_DIR, f'data_{SYMBOL_FILTER}_only.csv')
        df_filtered.to_csv(filename_filtered_csv, index=True if 'ts_recv' in df.columns else False)
        print(f"\nFiltered data saved as: {filename_filtered_csv}")

print(f"\n🎉 All outputs saved to: {OUTPUT_DIR}/")