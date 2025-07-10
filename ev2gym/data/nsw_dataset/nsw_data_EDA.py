import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import logging
from typing import Tuple, Optional
import pyarrow.parquet as pq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('nsw_data_eda.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12

def setup_output_directory() -> Path:
    """Create and return the output directory for EDA results."""
    output_dir = Path(__file__).parent / 'EDA'
    try:
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Output directory created at: {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

def load_external_features(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and return the external features dataset with enhanced debugging."""
    try:
        logger.info(f"Loading external features from: {file_path}")
        
        # Check if file exists and is not empty
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        if file_path.stat().st_size == 0:
            logger.error(f"File is empty: {file_path}")
            return None
            
        # Try to read metadata first
        try:
            meta = pq.read_metadata(file_path)
            logger.info(f"Parquet file metadata - Rows: {meta.num_rows}, Columns: {meta.num_columns}")
            logger.info(f"Schema: {meta.schema}")
        except Exception as e:
            logger.warning(f"Could not read parquet metadata: {e}")
        
        # Try to read the file
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully loaded {len(df)} rows")
            if len(df) == 0:
                logger.warning("Loaded an empty DataFrame")
                # Try to read with different engines
                for engine in ['pyarrow', 'fastparquet']:
                    try:
                        df = pd.read_parquet(file_path, engine=engine)
                        logger.info(f"Loaded {len(df)} rows using {engine} engine")
                        if len(df) > 0:
                            break
                    except Exception as e:
                        logger.warning(f"Failed to read with {engine} engine: {e}")
            return df
        except Exception as e:
            logger.error(f"Error reading parquet file: {e}")
            # Try alternative reading methods
            try:
                table = pq.read_table(file_path)
                df = table.to_pandas()
                logger.info(f"Read {len(df)} rows using pyarrow.Table")
                return df
            except Exception as e2:
                logger.error(f"Alternative read method also failed: {e2}")
                return None
    except Exception as e:
        logger.error(f"Unexpected error in load_external_features: {e}", exc_info=True)
        return None

def load_household_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load and return the household data."""
    try:
        logger.info(f"Loading household data from: {file_path}")
        # Try different encodings if needed
        for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                logger.info(f"Successfully loaded {len(df)} rows with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
        logger.error("Failed to load CSV with standard encodings")
        return None
    except Exception as e:
        logger.error(f"Error loading household data: {e}")
        return None

def preprocess_data(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Preprocess the input DataFrame."""
    try:
        logger.info(f"Preprocessing {dataset_name} data")
        
        # Convert timestamp columns
        time_cols = [col for col in df.columns if any(indicator in col.lower() 
                     for indicator in ['time', 'date', 'timestamp', 'interval'])]
        
        for col in time_cols:
            try:
                # If this is already a datetime column, ensure it's timezone-naive
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Converted {col} to datetime")
                
                # If this is the interval_start column, set it as index
                if col == 'interval_start':
                    df = df.set_index(col)
                    logger.info(f"Set {col} as index")
                    
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {e}")
        
        # Basic cleaning
        df = df.dropna(how='all')
        logger.info(f"After cleaning, {len(df)} rows remain")
        
        # Ensure index is a DatetimeIndex if it's not already
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime if it's not already
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                logger.info("Converted index to datetime")
            except Exception as e:
                logger.warning(f"Could not convert index to datetime: {e}")
        
        # Sort index to ensure proper time ordering
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()
            logger.info("Sorted DataFrame by datetime index")
        
        return df
    except Exception as e:
        logger.error(f"Error during preprocessing {dataset_name}: {e}")
        return None

def get_time_column(df: pd.DataFrame) -> str:
    """
    Helper function to identify the time column in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        str: Name of the time column
        
    Raises:
        ValueError: If no time column can be identified
    """
    # List of possible time column names to check
    time_indicators = ['time', 'date', 'timestamp', 'interval', 'datetime']
    
    # First, check for exact matches (case insensitive)
    for col in df.columns:
        if any(indicator in col.lower() for indicator in time_indicators):
            return col
    
    # If no column matches the patterns, check column dtypes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # If still not found, check if index is datetime
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.name or 'index'
    
    raise ValueError("Could not identify a time column in the DataFrame. "
                    "Please ensure your data includes a datetime column.")

def analyze_dataset(df: pd.DataFrame, dataset_name: str, output_dir: Path) -> None:
    """Perform EDA on a dataset and save visualizations."""
    if df is None or df.empty:
        logger.warning(f"No data available for {dataset_name} analysis")
        return
    
    try:
        logger.info(f"\n=== {dataset_name.upper()} Analysis ===")
        logger.info(f"Dataset shape: {df.shape}")
        
        # Basic statistics
        logger.info("\nSummary statistics:")
        logger.info(df.describe(include='all').to_string())
        
        # Missing values
        logger.info("\nMissing values:")
        logger.info(df.isnull().sum().to_string())
        
        # Time series analysis
        time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            time_col = time_cols[0]
            try:
                df = df.set_index(time_col)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                # Plot time series for each numeric column
                for col in numeric_cols:
                    try:
                        plt.figure()
                        df[col].resample('2W').mean().plot(title=f'Fortnightly {col} over Time')
                        plt.tight_layout()
                        output_path = output_dir / f'{dataset_name}_{col}_fortnightly.png'
                        plt.savefig(output_path)
                        plt.close()
                        logger.info(f"Saved plot: {output_path}")
                    except Exception as e:
                        logger.warning(f"Could not plot {col}: {e}")
                        
            except Exception as e:
                logger.error(f"Error in time series analysis: {e}")
    
    except Exception as e:
        logger.error(f"Error analyzing {dataset_name}: {e}")

def compare_datasets(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Compare attributes between external features and household data."""
    try:
        logger.info("\n=== Comparing Datasets ===")
        
        # Find common numeric columns
        ext_num_cols = ext_df.select_dtypes(include=[np.number]).columns
        hh_num_cols = household_df.select_dtypes(include=[np.number]).columns
        
        # Plot correlation heatmaps if there are numeric columns
        if len(ext_num_cols) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(ext_df[ext_num_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('External Features Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_dir / 'ext_correlation_heatmap.png')
            plt.close()
            logger.info("Saved external features correlation heatmap")
        
        if len(hh_num_cols) > 0:
            plt.figure(figsize=(12, 8))
            sns.heatmap(household_df[hh_num_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Household Data Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(output_dir / 'household_correlation_heatmap.png')
            plt.close()
            logger.info("Saved household data correlation heatmap")
    
    except Exception as e:
        logger.error(f"Error comparing datasets: {e}")

def plot_time_series_comparison(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot time series comparison between external features and household data."""
    try:
        logger.info("Generating time series comparison plots...")
        
        # Make copies to avoid modifying the original DataFrames
        ext_df = ext_df.copy()
        household_df = household_df.copy()
        
        # Ensure we have datetime indices
        if not isinstance(ext_df.index, pd.DatetimeIndex):
            time_col = get_time_column(ext_df)
            ext_df = ext_df.set_index(time_col)
        
        if not isinstance(household_df.index, pd.DatetimeIndex):
            time_col = get_time_column(household_df)
            household_df = household_df.set_index(time_col)
        
        # Sort indices to ensure proper time ordering
        ext_df = ext_df.sort_index()
        household_df = household_df.sort_index()
        
        # Resample to daily mean for better visualization
        ext_daily = ext_df.resample('D').mean()
        hh_daily = household_df.resample('D').mean()
        
        # Plot 1: Electricity Price vs Household Demand
        if 'RRP' in ext_daily.columns and 'demand' in hh_daily.columns:
            fig, ax1 = plt.subplots(figsize=(16, 8))
            
            color = 'tab:red'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('RRP (AUD/MWh)', color=color)
            ax1.plot(ext_daily.index, ext_daily['RRP'], color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:blue'
            ax2.set_ylabel('Household Demand (kWh)', color=color)
            ax2.plot(hh_daily.index, hh_daily['demand'], color=color, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=color)
            
            plt.title('Electricity Price vs Household Demand (Daily Averages)')
            fig.tight_layout()
            output_path = output_dir / 'price_vs_demand.png'
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        
        # Plot 2: Temperature vs Air Conditioning Usage
        if 'temp' in ext_daily.columns and 'air_con' in hh_daily.columns:
            fig, ax1 = plt.subplots(figsize=(16, 8))
            
            color = 'tab:orange'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Temperature (°C)', color=color)
            ax1.plot(ext_daily.index, ext_daily['temp'], color=color, alpha=0.7)
            ax1.tick_params(axis='y', labelcolor=color)
            
            ax2 = ax1.twinx()
            color = 'tab:green'
            ax2.set_ylabel('Air Conditioning Usage (kWh)', color=color)
            ax2.plot(hh_daily.index, hh_daily['air_con'], color=color, alpha=0.7)
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 0.008)  # Set y-axis limit for AC power
            
            plt.title('Temperature vs Air Conditioning Usage (Daily Averages)')
            fig.tight_layout()
            output_path = output_dir / 'temp_vs_aircon.png'
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        
        # Plot 3: Energy Mix Components
        if all(col in ext_daily.columns for col in ['Solar', 'Wind', 'Natural Gas (Pipeline)']):
            plt.figure(figsize=(16, 8))
            
            # Resample to weekly for cleaner visualization
            ext_weekly = ext_daily[['Solar', 'Wind', 'Natural Gas (Pipeline)']].resample('W').mean()
            
            plt.stackplot(ext_weekly.index, 
                         ext_weekly['Solar'], 
                         ext_weekly['Wind'], 
                         ext_weekly['Natural Gas (Pipeline)'],
                         labels=['Solar', 'Wind', 'Natural Gas'])
            
            plt.title('Weekly Average Energy Mix (MW)')
            plt.legend(loc='upper left')
            plt.tight_layout()
            output_path = output_dir / 'energy_mix_weekly.png'
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved plot: {output_path}")
            
    except Exception as e:
        logger.error(f"Error in time series comparison: {e}", exc_info=True)

def analyze_correlations(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze correlations between external features and household data."""
    try:
        logger.info("Analyzing cross-dataset correlations...")
        
        # Make copies to avoid modifying the original DataFrames
        ext_df = ext_df.copy()
        household_df = household_df.copy()
        
        # Ensure we have datetime indices
        if not isinstance(ext_df.index, pd.DatetimeIndex):
            time_col = get_time_column(ext_df)
            ext_df = ext_df.set_index(time_col)
        
        if not isinstance(household_df.index, pd.DatetimeIndex):
            time_col = get_time_column(household_df)
            household_df = household_df.set_index(time_col)
        
        # Sort indices to ensure proper alignment
        ext_df = ext_df.sort_index()
        household_df = household_df.sort_index()
        
        # Resample to daily mean for correlation analysis
        ext_daily = ext_df.resample('D').mean()
        hh_daily = household_df.resample('D').mean()
        
        # Ensure we have overlapping data
        start_date = max(ext_daily.index.min(), hh_daily.index.min())
        end_date = min(ext_daily.index.max(), hh_daily.index.max())
        
        if start_date > end_date:
            logger.warning("No overlapping dates between external features and household data")
            return
            
        ext_daily = ext_daily[start_date:end_date]
        hh_daily = hh_daily[start_date:end_date]
        
        # Combine datasets for correlation analysis
        combined = pd.concat([ext_daily, hh_daily], axis=1).dropna()
        
        if len(combined) == 0:
            logger.warning("No overlapping data after aligning and dropping NA values")
            return
            
        # Calculate correlations
        corr_matrix = combined.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
        plt.title('Correlation Between External Features and Household Usage')
        plt.tight_layout()
        output_path = output_dir / 'cross_dataset_correlation.png'
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved correlation heatmap: {output_path}")
        
        # Print top correlations with household metrics
        household_cols = [col for col in hh_daily.columns if col in combined.columns]
        for col in household_cols:
            correlations = corr_matrix[col].drop(household_cols, errors='ignore')
            if not correlations.empty:
                top_corrs = correlations.abs().sort_values(ascending=False).head(5)
                logger.info(f"\nTop correlations with {col}:")
                for feature, corr in top_corrs.items():
                    logger.info(f"  {feature}: {corr:.3f}")
        
    except Exception as e:
        logger.error(f"Error in correlation analysis: {e}", exc_info=True)

def plot_daily_patterns(household_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot average daily consumption patterns."""
    try:
        logger.info("Generating daily consumption patterns...")
        
        # Extract hour from datetime index
        household_df['hour'] = household_df.index.hour
        
        # Group by hour and calculate mean
        hourly_avg = household_df.groupby('hour').mean()
        
        # Plot
        plt.figure(figsize=(12, 6))
        for col in ['demand', 'solar', 'hot_water']:
            if col in hourly_avg.columns:
                plt.plot(hourly_avg.index, hourly_avg[col], label=col)
        
        plt.title('Average Daily Energy Consumption Patterns')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Energy (kWh)')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        output_path = output_dir / 'daily_patterns.png'
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved plot: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in daily patterns analysis: {e}", exc_info=True)

def plot_seasonal_analysis(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot seasonal energy consumption patterns."""
    try:
        logger.info("Generating seasonal analysis plots...")
        
        # Extract season from datetime index
        seasons = {1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 
                  6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn', 11: 'Autumn', 12: 'Winter'}
        
        # Add season to both dataframes
        ext_df['season'] = ext_df.index.month.map(seasons)
        household_df['season'] = household_df.index.month.map(seasons)
        
        # Group by season and calculate mean
        ext_seasonal = ext_df.groupby('season').mean()
        hh_seasonal = household_df.groupby('season').mean()
        
        # Plot external features by season
        plt.figure(figsize=(12, 8))
        ext_seasonal[['RRP', 'temp']].plot(kind='bar', subplots=True, layout=(1,2))
        plt.suptitle('Seasonal Variation in External Features')
        plt.tight_layout()
        output_path = output_dir / 'seasonal_external.png'
        plt.savefig(output_path)
        plt.close()
        
        # Plot household consumption by season
        plt.figure(figsize=(12, 8))
        hh_seasonal[['demand', 'solar', 'hot_water']].plot(kind='bar')
        plt.title('Seasonal Variation in Household Energy Consumption')
        plt.ylabel('Average Energy (kWh)')
        plt.tight_layout()
        output_path = output_dir / 'seasonal_household.png'
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved seasonal analysis plots")
        
    except Exception as e:
        logger.error(f"Error in seasonal analysis: {e}", exc_info=True)

def plot_renewable_contribution(ext_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot renewable energy contribution over time."""
    try:
        logger.info("Generating renewable energy contribution plot...")
        
        # Calculate renewable percentage
        renewable_cols = ['Solar', 'Wind', 'Hydro']
        fossil_cols = ['Black coal', 'Natural Gas (Pipeline)']
        
        ext_df['renewable'] = ext_df[renewable_cols].sum(axis=1)
        ext_df['fossil'] = ext_df[fossil_cols].sum(axis=1)
        ext_df['renewable_pct'] = ext_df['renewable'] / (ext_df['renewable'] + ext_df['fossil']) * 100
        
        # Resample to weekly
        weekly_renewable = ext_df['renewable_pct'].resample('W').mean()
        
        # Plot
        plt.figure(figsize=(12, 6))
        weekly_renewable.plot()
        plt.title('Weekly Renewable Energy Contribution')
        plt.ylabel('Renewable Percentage (%)')
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        output_path = output_dir / 'renewable_contribution.png'
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved plot: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in renewable contribution analysis: {e}", exc_info=True)

def plot_monthly_comparison(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot monthly comparison of key attributes."""
    try:
        logger.info("Generating monthly comparison plots...")
        
        # Create monthly resampled data
        ext_monthly = ext_df.resample('M').mean(numeric_only=True)
        hh_monthly = household_df.resample('M').mean(numeric_only=True)
        
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Electricity Price and Demand
        ax1 = axs[0, 0]
        color = 'tab:red'
        ax1.set_xlabel('Month')
        ax1.set_ylabel('RRP (AUD/MWh)', color=color)
        ax1.plot(ext_monthly.index, ext_monthly['RRP'], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1b = ax1.twinx()
        color = 'tab:blue'
        ax1b.set_ylabel('Household Demand (kWh)', color=color)
        ax1b.plot(hh_monthly.index, hh_monthly['demand'], color=color, marker='s')
        ax1b.tick_params(axis='y', labelcolor=color)
        ax1.set_title('Monthly Electricity Price vs Household Demand')
        
        # Plot 2: Temperature and Hot Water Usage
        ax2 = axs[0, 1]
        color = 'tab:orange'
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Temperature (°C)', color=color)
        ax2.plot(ext_monthly.index, ext_monthly['temp'], color=color, marker='o')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax2b = ax2.twinx()
        color = 'tab:green'
        ax2b.set_ylabel('Hot Water Usage (kWh)', color=color)
        ax2b.plot(hh_monthly.index, hh_monthly['hot_water'], color=color, marker='s')
        ax2b.tick_params(axis='y', labelcolor=color)
        ax2.set_title('Monthly Temperature vs Hot Water Usage')
        
        # Plot 3: Renewable Energy and Solar Generation
        ax3 = axs[1, 0]
        color = 'tab:purple'
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Renewable Percentage (%)', color=color)
        # Calculate renewable percentage if not already present
        if 'renewable_pct' not in ext_monthly.columns:
            renewable_cols = ['Solar', 'Wind', 'Hydro']
            fossil_cols = ['Black coal', 'Natural Gas (Pipeline)']
            ext_monthly['renewable'] = ext_monthly[renewable_cols].sum(axis=1)
            ext_monthly['fossil'] = ext_monthly[fossil_cols].sum(axis=1)
            ext_monthly['renewable_pct'] = ext_monthly['renewable'] / (ext_monthly['renewable'] + ext_monthly['fossil']) * 100
        ax3.plot(ext_monthly.index, ext_monthly['renewable_pct'], color=color, marker='o')
        ax3.tick_params(axis='y', labelcolor=color)
        
        ax3b = ax3.twinx()
        color = 'tab:brown'
        ax3b.set_ylabel('Solar Generation (kWh)', color=color)
        # Use absolute value since solar is negative (generation)
        ax3b.plot(hh_monthly.index, -hh_monthly['solar'], color=color, marker='s')
        ax3b.tick_params(axis='y', labelcolor=color)
        ax3.set_title('Monthly Renewable Energy vs Solar Generation')
        
        # Plot 4: CO2 Emissions and Total Generation
        ax4 = axs[1, 1]
        color = 'tab:cyan'
        ax4.set_xlabel('Month')
        ax4.set_ylabel('CO2 Emissions (kg/MWh)', color=color)
        ax4.plot(ext_monthly.index, ext_monthly['Total_CO2_Emissions'], color=color, marker='o')
        ax4.tick_params(axis='y', labelcolor=color)
        
        ax4b = ax4.twinx()
        color = 'tab:olive'
        ax4b.set_ylabel('Total Generation (MW)', color=color)
        ax4b.plot(ext_monthly.index, ext_monthly['Total_Generation'], color=color, marker='s')
        ax4b.tick_params(axis='y', labelcolor=color)
        ax4.set_title('Monthly CO2 Emissions vs Total Generation')
        
        plt.tight_layout()
        output_path = output_dir / 'monthly_comparison.png'
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved plot: {output_path}")
        
    except Exception as e:
        logger.error(f"Error in monthly comparison analysis: {e}", exc_info=True)

def plot_weekly_relationships_by_week(ext_df: pd.DataFrame, household_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot weekly relationships between different attributes for each week separately."""
    try:
        logger.info("Generating weekly relationships plots by week...")
        
        # Create a week identifier (year_week) for grouping
        household_df['year_week'] = household_df.index.isocalendar().year.astype(str) + '_' + household_df.index.isocalendar().week.astype(str)
        weeks = household_df['year_week'].unique()
        
        # List of relationships to plot: (attribute1, attribute2, label1, label2, color1, color2)
        relationships = [
            ('demand', 'RRP', 'Household Demand (kWh)', 'Electricity Price (AUD/MWh)', 'tab:blue', 'tab:red'),
            ('demand', 'temp', 'Household Demand (kWh)', 'Temperature (°C)', 'tab:blue', 'tab:green'),
            ('solar', 'RRP', 'Household Solar (kWh)', 'Electricity Price (AUD/MWh)', 'tab:orange', 'tab:red'),
            ('solar', 'temp', 'Household Solar (kWh)', 'Temperature (°C)', 'tab:orange', 'tab:green'),
            ('air_con', 'RRP', 'Air Conditioning (kWh)', 'Electricity Price (AUD/MWh)', 'tab:purple', 'tab:red'),
            ('air_con', 'temp', 'Air Conditioning (kWh)', 'Temperature (°C)', 'tab:purple', 'tab:green'),
        ]
        
        for week in weeks:
            # Get household data for the week
            hh_week = household_df[household_df['year_week'] == week]
            if hh_week.empty:
                continue
                
            # Get the time range for the week
            start = hh_week.index.min()
            end = hh_week.index.max()
            
            # Get external data for the same week
            ext_week = ext_df[start:end]
            
            # Create a figure with subplots for each relationship
            n_rels = len(relationships)
            fig, axes = plt.subplots(n_rels, 1, figsize=(18, 4 * n_rels))
            
            # If only one subplot, axes is not a list
            if n_rels == 1:
                axes = [axes]
            
            for i, (attr1, attr2, label1, label2, color1, color2) in enumerate(relationships):
                ax1 = axes[i]
                ax1.set_xlabel('Time')
                ax1.set_ylabel(label1, color=color1)
                
                # Check if attr1 is in household or external
                if attr1 in hh_week.columns:
                    ax1.plot(hh_week.index, hh_week[attr1], color=color1, label=label1)
                elif attr1 in ext_week.columns:
                    ax1.plot(ext_week.index, ext_week[attr1], color=color1, label=label1, marker='o', markersize=3, linestyle='-')
                else:
                    # Skip if not found
                    logger.warning(f"Attribute {attr1} not found in week {week}")
                    continue
                ax1.tick_params(axis='y', labelcolor=color1)
                
                ax2 = ax1.twinx()
                ax2.set_ylabel(label2, color=color2)
                if attr2 in hh_week.columns:
                    ax2.plot(hh_week.index, hh_week[attr2], color=color2, label=label2)
                elif attr2 in ext_week.columns:
                    ax2.plot(ext_week.index, ext_week[attr2], color=color2, label=label2, marker='o', markersize=3, linestyle='-')
                else:
                    logger.warning(f"Attribute {attr2} not found in week {week}")
                    continue
                ax2.tick_params(axis='y', labelcolor=color2)
                
                # Set y-axis limit for AC power
                if attr1 == 'air_con':
                    ax1.set_ylim(0, 0.008)
                
                ax1.set_title(f'Week {week}: {label1} vs {label2}')
            
            plt.tight_layout()
            output_path = output_dir / f'weekly_relationships_week_{week}.png'
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved plot: {output_path}")
        
        # Remove the temporary column
        household_df.drop(columns=['year_week'], inplace=True)
        
    except Exception as e:
        logger.error(f"Error in weekly relationships analysis by week: {e}", exc_info=True)

def main():
    try:
        logger.info("Starting NSW Dataset EDA...")
        
        # Setup output directory
        output_dir = setup_output_directory()
        
        # Define file paths
        base_dir = Path(__file__).parent
        ext_path = base_dir / 'external_features' / 'external_dataset.parquet'
        household_path = base_dir / 'households' / 'SuRF-property_32.csv'
        
        # Check if files exist
        if not ext_path.exists():
            logger.error(f"External features file not found at: {ext_path}")
            logger.info("Available files in directory:")
            for f in (base_dir / 'external_features').glob('*'):
                logger.info(f"- {f.name} ({f.stat().st_size} bytes)")
        
        # Load and preprocess data
        ext_df = load_external_features(ext_path)
        household_df = load_household_data(household_path)
        
        # If external features failed to load, try to find alternative files
        if ext_df is None or len(ext_df) == 0:
            logger.warning("Trying to find alternative external feature files...")
            parquet_files = list((base_dir / 'external_features').glob('*.parquet'))
            if parquet_files:
                logger.info(f"Found {len(parquet_files)} parquet files. Trying each one...")
                for p_file in parquet_files:
                    logger.info(f"Trying file: {p_file}")
                    ext_df = load_external_features(p_file)
                    if ext_df is not None and len(ext_df) > 0:
                        logger.info(f"Successfully loaded data from {p_file}")
                        break
        
        if ext_df is not None and len(ext_df) > 0:
            ext_df = preprocess_data(ext_df, "external_features")
            analyze_dataset(ext_df, "external_features", output_dir)
        else:
            logger.error("Could not load any external features data")
        
        if household_df is not None and len(household_df) > 0:
            household_df = preprocess_data(household_df, "household")
            analyze_dataset(household_df, "household", output_dir)
            
            # Compare datasets if both are available
            if ext_df is not None and len(ext_df) > 0:
                compare_datasets(ext_df, household_df, output_dir)
                plot_time_series_comparison(ext_df, household_df, output_dir)
                analyze_correlations(ext_df, household_df, output_dir)
                plot_daily_patterns(household_df, output_dir)
                plot_seasonal_analysis(ext_df, household_df, output_dir)
                plot_renewable_contribution(ext_df, output_dir)
                plot_monthly_comparison(ext_df, household_df, output_dir)
                plot_weekly_relationships_by_week(ext_df, household_df, output_dir)
        
        logger.info(f"\nEDA complete! Results saved to: {output_dir}")
        
    except Exception as e:
        logger.critical(f"Critical error in main: {e}", exc_info=True)

if __name__ == "__main__":
    main()
