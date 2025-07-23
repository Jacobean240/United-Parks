import pandas as pd
import numpy as np

def distribute_integers(total, shares):
    """Distribute integer total proportionally to shares, ensuring sum equals total"""
    if total == 0 or shares.sum() == 0:
        return np.zeros(len(shares), dtype=int)
    
    # Calculate proportional amounts
    proportions = shares / shares.sum()
    exact_amounts = total * proportions
    
    # Use largest remainder method to distribute integers
    integer_parts = np.floor(exact_amounts).astype(int)
    remainders = exact_amounts - integer_parts
    
    # Distribute remaining tickets to largest remainders
    remaining = total - integer_parts.sum()
    if remaining > 0:
        largest_remainders = np.argsort(remainders)[-remaining:]
        integer_parts[largest_remainders] += 1
    
    return integer_parts

def main():
    print("Loading data...")
    
    # Load both files
    sales_df = pd.read_csv('Combine Data/Weekly_Paid_Media_and_Sales_Data.xlsx - Product Sales (1).csv')
    media_df = pd.read_csv('Combine Data/United Parks - Paid Media - Sheet1.csv')
    
    # Convert dates
    sales_df['Week'] = pd.to_datetime(sales_df['Week'])
    media_df['Week'] = pd.to_datetime(media_df['Week'])
    
    print("Distributing product sales across paid media...")
    
    # Merge with sales data first
    combined_df = media_df.merge(sales_df[['Week', 'Park', 'Single-Day Tickets', 'Annual Passes', 'Reservations']], 
                                on=['Week', 'Park'], how='left')
    
    # Initialize distributed columns
    combined_df['Distributed_Single_Day'] = 0
    combined_df['Distributed_Annual_Passes'] = 0
    combined_df['Distributed_Reservations'] = 0
    
    # Process each week/park combination separately
    for (week, park), group in combined_df.groupby(['Week', 'Park']):
        indices = group.index
        conversions = group['Conversions'].values
        
        # Get original totals for this week/park
        single_day_total = group['Single-Day Tickets'].iloc[0]
        annual_pass_total = group['Annual Passes'].iloc[0]
        reservation_total = group['Reservations'].iloc[0]
        
        # Distribute using proper integer distribution
        combined_df.loc[indices, 'Distributed_Single_Day'] = distribute_integers(single_day_total, conversions)
        combined_df.loc[indices, 'Distributed_Annual_Passes'] = distribute_integers(annual_pass_total, conversions)
        combined_df.loc[indices, 'Distributed_Reservations'] = distribute_integers(reservation_total, conversions)
    
    # Replace original columns with distributed values
    combined_df['Single-Day Tickets'] = combined_df['Distributed_Single_Day']
    combined_df['Annual Passes'] = combined_df['Distributed_Annual_Passes']
    combined_df['Reservations'] = combined_df['Distributed_Reservations']
    
    # Remove working columns
    combined_df = combined_df.drop(['Distributed_Single_Day', 'Distributed_Annual_Passes', 'Distributed_Reservations'], axis=1)
    
    # Save combined file
    combined_df.to_csv('Combine Data/combined_paid_media_sales.csv', index=False)
    
    print(f"✅ Combined data saved: {len(combined_df)} rows")
    print("File: Combine Data/combined_paid_media_sales.csv")
    
    # Validation check
    print("\nValidation check:")
    original_totals = sales_df.groupby('Week')[['Single-Day Tickets', 'Annual Passes', 'Reservations']].sum()
    distributed_totals = combined_df.groupby('Week')[['Single-Day Tickets', 'Annual Passes', 'Reservations']].sum()
    
    print("Original vs Distributed totals match:", 
          (original_totals == distributed_totals).all().all())
    
    print("\nSample preview:")
    print(combined_df[['Week', 'Park', 'Channel', 'Platform', 'Conversions', 'Single-Day Tickets', 'Annual Passes', 'Reservations']].head())

if __name__ == "__main__":
    main() 