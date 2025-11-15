import os
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio

# Wczytanie danych
df = pd.read_csv('databentofeature_30T_1_test_1.csv', low_memory=False)

print("Pierwsze wiersze danych przed filtrowaniem:")
print(df.head())
print(f"\nLiczba kolumn przed filtrowaniem: {len(df.columns)}")
print(f"Kolumny przed filtrowaniem: {list(df.columns)}")

# FILTROWANIE KOLUMN ZAWIERAJĄCYCH "6EH5" W NAZWIE
columns_with_6EH5 = [col for col in df.columns if '6EH5' in col]
print(f"\nKolumny zawierające '6EH5': {columns_with_6EH5}")
print(f"Liczba kolumn z '6EH5': {len(columns_with_6EH5)}")

# Tworzenie nowego DataFrame tylko z kolumnami zawierającymi "6EH5"
df_filtered = df[columns_with_6EH5].copy()

print(f"\nDataFrame po filtrowaniu - kształt: {df_filtered.shape}")
print("Pierwsze wiersze po filtrowaniu:")
print(df_filtered.head())

# Sprawdzenie czy są jakieś kolumny po filtrowaniu
if len(columns_with_6EH5) == 0:
    print("\n⚠️  BRAK KOLUMN Z '6EH5' W NAZWIE!")
    print("Sprawdź czy nazwa instrumentu jest poprawna.")
else:
    # Kontynuacja przetwarzania tylko jeśli są kolumny z "6EH5"
    print("\n" + "="*60)
    print("PRZETWARZANIE DANYCH Z KOLUMNAMI 6EH5")
    print("="*60)
    
    # Konwersja kolumny 'ts_recv' jeśli istnieje w oryginalnym df
    if 'ts_recv' in df.columns:
        df_filtered.index = pd.to_datetime(df['ts_recv'])
        # Usunięcie timezone
        df_filtered.index = df_filtered.index.tz_localize(None)
        print("Ustawiono index na podstawie 'ts_recv'")
    else:
        print("Kolumna 'ts_recv' nie istnieje - pomijam ustawianie indexu")

    print(f"\nDane po filtrowaniu i ustawieniu indexu:")
    print(df_filtered.head())

    # Sprawdzenie typów danych
    print("\nTypy danych w kolumnach po filtrowaniu:")
    print(df_filtered.dtypes)

    # Wybierz tylko kolumny numeryczne
    numeric_columns = df_filtered.select_dtypes(include=['number']).columns
    print(f"\nKolumny numeryczne po filtrowaniu: {len(numeric_columns)}")
    print(f"Kolumny numeryczne: {list(numeric_columns)}")

    if len(numeric_columns) == 0:
        print("Brak kolumn numerycznych w danych po filtrowaniu!")
    else:
        # Obliczenie macierzy korelacji tylko dla kolumn numerycznych
        corr_matrix = df_filtered[numeric_columns].corr()

        print(f"\nRozmiar macierzy korelacji: {corr_matrix.shape}")
        
        # OBLICZENIE ŚREDNIEJ KORELACJI
        corr_values = corr_matrix.values
        upper_triangle = corr_values[np.triu_indices_from(corr_values, k=1)]
        
        mean_correlation = np.mean(upper_triangle)
        mean_abs_correlation = np.mean(np.abs(upper_triangle))
        
        print(f"\nŚREDNIA KORELACJA: {mean_correlation:.4f}")
        print(f"ŚREDNIA |KORELACJA|: {mean_abs_correlation:.4f}")
        print(f"Zakres korelacji: [{np.min(upper_triangle):.4f}, {np.max(upper_triangle):.4f}]")
        print(f"Liczba analizowanych par: {len(upper_triangle)}")

        # WIZUALIZACJA MATPLOTLIB/SEABORN (statyczna)
        print("\n" + "="*50)
        print("TWORZENIE WIZUALIZACJI STATYCZNEJ")
        print("="*50)
        
        plt.style.use('default')
        plt.figure(figsize=(12, 10))

        # Tworzenie heatmapy
        sb.heatmap(corr_matrix, 
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   linewidths=0.5,
                   linecolor='white',
                   square=True)

        plt.title('Macierz korelacji cech 6EH5', fontsize=16, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Zapis statycznego wykresu do pliku PNG
        filename_static = 'macierz_korelacji_6EH5_static.png'
        plt.savefig(filename_static, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Statyczny wykres zapisany jako: {filename_static}")
        
        plt.show()

        # WIZUALIZACJA PLOTLY (interaktywna)
        print("\n" + "="*50)
        print("TWORZENIE WIZUALIZACJI INTERAKTYWNEJ")
        print("="*50)
        
        # Interaktywna heatmapa
        fig = px.imshow(corr_matrix, 
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title=f'Macierz korelacji cech 6EH5 (Średnia: {mean_correlation:.3f})',
                        labels=dict(color="Korelacja"))

        # Dostosowanie layoutu
        fig.update_layout(
            width=1200, 
            height=1000,
            xaxis_title="Cechy 6EH5",
            yaxis_title="Cechy 6EH5",
            font=dict(size=10)
        )
        
        # Dostosowanie hover info
        fig.update_traces(
            hovertemplate="<br>".join([
                "Cecha X: %{x}",
                "Cecha Y: %{y}", 
                "Korelacja: %{z:.3f}",
                "<extra></extra>"
            ])
        )
        
        # Zapis interaktywnego wykresu
        filename_interactive = 'macierz_korelacji_6EH5_interactive.html'
        pio.write_html(fig, filename_interactive)
        print(f"Interaktywny wykres zapisany jako: {filename_interactive}")
        
        # Pokazanie interaktywnego wykresu
        try:
            fig.show()
        except Exception as e:
            print(f"Uwaga: Nie udało się wyświetlić interaktywnego wykresu: {e}")

        # Zapis przefiltrowanych danych do nowego pliku CSV
        filename_filtered_csv = 'dane_6EH5_only.csv'
        df_filtered.to_csv(filename_filtered_csv, index=True if 'ts_recv' in df.columns else False)
        print(f"\nPrzefiltrowane dane zapisane jako: {filename_filtered_csv}")