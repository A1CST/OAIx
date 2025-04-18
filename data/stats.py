import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Path to the survival log file
SURVIVAL_LOG_PATH = os.path.join(os.path.dirname(__file__), 'survival_log.csv')

def read_survival_log():
    """Read the survival log CSV file and return a pandas DataFrame."""
    try:
        # First, peek at the file to see if it has a header
        with open(SURVIVAL_LOG_PATH, 'r') as f:
            first_line = f.readline().strip()
        
        # Check if the first line looks like a header
        if 'day' in first_line.lower() and 'ticks' in first_line.lower():
            # File has a header, read with header=0
            df = pd.read_csv(SURVIVAL_LOG_PATH, 
                            dtype={'day': int, 'ticks': int, 'food': int, 'death_cause': str})
            print("Reading CSV with header")
        else:
            # No header, use our column names
            df = pd.read_csv(SURVIVAL_LOG_PATH, header=None, 
                            names=['day', 'ticks', 'food', 'death_cause'],
                            dtype={'day': int, 'ticks': int, 'food': int, 'death_cause': str})
            print("Reading CSV without header")
            
        return df
    except FileNotFoundError:
        print(f"File not found: {SURVIVAL_LOG_PATH}")
        return pd.DataFrame(columns=['day', 'ticks', 'food', 'death_cause'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        # If there's an error, try a more robust approach
        try:
            # Try reading with pandas auto-detection
            df = pd.read_csv(SURVIVAL_LOG_PATH)
            
            # Check if we need to rename columns
            columns = list(df.columns)
            if len(columns) == 4:
                # If the columns don't match our expected names, rename them
                column_mapping = {
                    columns[0]: 'day',
                    columns[1]: 'ticks',
                    columns[2]: 'food',
                    columns[3]: 'death_cause'
                }
                df = df.rename(columns=column_mapping)
                
            # Convert columns to appropriate types
            df['day'] = pd.to_numeric(df['day'], errors='coerce')
            df['ticks'] = pd.to_numeric(df['ticks'], errors='coerce')
            df['food'] = pd.to_numeric(df['food'], errors='coerce')
            
            # Fill NaN values
            df = df.fillna({'day': 0, 'ticks': 0, 'food': 0, 'death_cause': 'unknown'})
            
            # Convert to integer types
            df['day'] = df['day'].astype(int)
            df['ticks'] = df['ticks'].astype(int)
            df['food'] = df['food'].astype(int)
            
            print(f"Successfully recovered data with columns: {df.columns.tolist()}")
            return df
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            return pd.DataFrame(columns=['day', 'ticks', 'food', 'death_cause'])

def format_tick_labels(value, pos):
    """Format large tick values to be more readable (K for thousands)."""
    # Make sure value is numeric, convert if it's not
    try:
        value_int = int(value)
        if value_int >= 1000:
            return f'{value_int//1000}K'
        return str(value_int)
    except (ValueError, TypeError):
        return str(value)

def update_plot(df):
    """Update the matplotlib charts with the latest data ensuring proper y-axis ticks."""
    # Print data summary for debugging
    print(f"Data shape: {df.shape}")
    print(f"Column types: {df.dtypes}")
    print(f"Max ticks value: {df['ticks'].max()}")
    
    plt.clf()
    
    # Create subplots with better spacing
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [1, 1, 0.8]})
    
    # Plot survival time (ticks) per game with improved y-axis
    if not df.empty:
        ax1.plot(df['day'], df['ticks'], 'b-', marker='o', markersize=3, label='Survival Time')
        ax1.set_title('Survival Time per Day', fontsize=14)
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Ticks', fontsize=12)
        
        # Calculate appropriate y-axis limits and ticks
        max_ticks = df['ticks'].max()
        if max_ticks > 0:
            # Set a reasonable number of tick points
            y_max = max_ticks * 1.1  # Add 10% padding
            
            # Create tick formatter for better readability
            formatter = FuncFormatter(format_tick_labels)
            ax1.yaxis.set_major_formatter(formatter)
            
            # Force integer ticks and set reasonable number of ticks
            ax1.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax1.set_ylim(0, y_max)
        
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='upper right')
        
        # Add moving average trendline if we have enough data
        if len(df) > 5:
            window_size = min(5, len(df) // 2)  # Use either 5 or half the data points, whichever is smaller
            moving_avg = df['ticks'].rolling(window=window_size).mean()
            ax1.plot(df['day'][window_size-1:], moving_avg[window_size-1:], 
                     'r-', linewidth=2, label=f'{window_size}-Day Moving Avg')
            ax1.legend()
    
    # Plot food eaten per day
    if not df.empty:
        ax2.plot(df['day'], df['food'], 'g-', marker='o', markersize=3, label='Food Eaten')
        ax2.set_title('Food Eaten per Day', fontsize=14)
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Food Items', fontsize=12)
        
        # Set appropriate y-axis for food
        max_food = df['food'].max()
        if max_food > 0:
            y_max_food = max_food * 1.1  # Add 10% padding
            ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
            ax2.set_ylim(0, y_max_food)
        
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc='upper right')
        
        # Add moving average for food as well
        if len(df) > 5:
            window_size = min(5, len(df) // 2)
            food_moving_avg = df['food'].rolling(window=window_size).mean()
            ax2.plot(df['day'][window_size-1:], food_moving_avg[window_size-1:], 
                     'r-', linewidth=2, label=f'{window_size}-Day Moving Avg')
            ax2.legend()
    
    # Plot death reasons with custom colors
    if not df.empty:
        death_counts = df['death_cause'].value_counts()
        
        # Define colors for each death reason
        death_colors = {
            'enemy': 'red',
            'starvation': 'orange',
            'wall': 'purple',
            'exhaustion': 'blue',
            'unknown': 'gray'
        }
        
        # Get colors for each category, use default if not defined
        colors = [death_colors.get(reason, 'blue') for reason in death_counts.index]
        
        bars = ax3.bar(death_counts.index, death_counts.values, color=colors)
        ax3.set_title('Death Causes Distribution', fontsize=14)
        ax3.set_xlabel('Death Cause', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        
        # Add count labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add timestamp to the figure
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.01, f'Last Updated: {current_time}', ha='center', fontsize=10)
    
    # Add data summary
    if not df.empty:
        total_ticks = df['ticks'].sum()
        avg_ticks = df['ticks'].mean()
        total_food = df['food'].sum()
        
        summary_text = (
            f"Total Simulated Time: {format_tick_labels(total_ticks, 0)} ticks\n"
            f"Avg Survival: {int(avg_ticks)} ticks\n"
            f"Total Food Eaten: {total_food}"
        )
        fig.text(0.02, 0.01, summary_text, ha='left', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.97])
    plt.savefig('survival_stats.png', dpi=120)
    plt.close(fig)
    print("Plot saved to survival_stats.png")

def main():
    """Main function to monitor the survival log and update the plot."""
    print("Starting survival log monitor...")
    last_modified = 0
    
    while True:
        try:
            # Check if the file exists
            if not os.path.exists(SURVIVAL_LOG_PATH):
                print(f"Waiting for log file to be created at: {SURVIVAL_LOG_PATH}")
                time.sleep(10)
                continue
                
            # Check if the file has been modified
            current_modified = os.path.getmtime(SURVIVAL_LOG_PATH)
            
            if current_modified > last_modified or last_modified == 0:
                # Read and update the plot
                df = read_survival_log()
                if not df.empty:
                    update_plot(df)
                    last_modified = current_modified
                    print(f"Updated plot with {len(df)} days of data")
                else:
                    print("Log file exists but contains no data.")
            
            # Wait before checking again
            time.sleep(30)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    main()