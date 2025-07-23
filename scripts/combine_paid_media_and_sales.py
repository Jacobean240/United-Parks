import pandas as pd
import numpy as np
from datetime import datetime
import os

def load_and_clean_data():
    """Load and clean both datasets"""
    print("Loading datasets...")
    
    # Load product sales data
    try:
        sales_df = pd.read_csv('Combine Data/Weekly_Paid_Media_and_Sales_Data.xlsx - Product Sales (1).csv')
        print(f"Product sales data loaded: {len(sales_df)} rows")
    except FileNotFoundError:
        print("Error: Product sales CSV file not found")
        return None, None
    
    # Load paid media data  
    try:
        media_df = pd.read_csv('Combine Data/United Parks - Paid Media - Sheet1.csv')
        print(f"Paid media data loaded: {len(media_df)} rows")
    except FileNotFoundError:
        print("Error: Paid media CSV file not found")
        return None, None
    
    # Clean sales data - remove empty columns
    sales_df = sales_df.dropna(axis=1, how='all')
    
    # Convert date columns to datetime
    sales_df['Week'] = pd.to_datetime(sales_df['Week'])
    media_df['Week'] = pd.to_datetime(media_df['Week'])
    
    # Clean any NaN values in key columns
    sales_df = sales_df.dropna(subset=['Week', 'Park'])
    media_df = media_df.dropna(subset=['Week', 'Park', 'Conversions'])
    
    print("Data cleaning completed.")
    return sales_df, media_df

def calculate_conversion_weights(media_df):
    """Calculate the weight of each paid media conversion relative to total conversions per week/park"""
    print("Calculating conversion weights...")
    
    # Group by Week and Park to get total conversions
    park_totals = media_df.groupby(['Week', 'Park'])['Conversions'].sum().reset_index()
    park_totals.rename(columns={'Conversions': 'Total_Conversions'}, inplace=True)
    
    # Merge back to get total conversions for each row
    media_weighted = media_df.merge(park_totals, on=['Week', 'Park'], how='left')
    
    # Calculate conversion weight (what percentage of total conversions this row represents)
    media_weighted['Conversion_Weight'] = media_weighted['Conversions'] / media_weighted['Total_Conversions']
    
    # Handle division by zero (when total conversions = 0)
    media_weighted['Conversion_Weight'] = media_weighted['Conversion_Weight'].fillna(0)
    
    print("Conversion weights calculated.")
    return media_weighted

def distribute_ticket_sales(sales_df, media_weighted):
    """Distribute actual ticket sales across paid media conversions based on conversion weights"""
    print("Distributing ticket sales across paid media conversions...")
    
    # Merge sales data with weighted media data
    combined_df = media_weighted.merge(sales_df[['Week', 'Park', 'Single-Day Tickets', 'Annual Passes', 'Reservations']], 
                                      on=['Week', 'Park'], how='left')
    
    # Calculate distributed ticket sales for each paid media row
    combined_df['Distributed_Single_Day'] = combined_df['Single-Day Tickets'] * combined_df['Conversion_Weight']
    combined_df['Distributed_Annual_Passes'] = combined_df['Annual Passes'] * combined_df['Conversion_Weight'] 
    combined_df['Distributed_Reservations'] = combined_df['Reservations'] * combined_df['Conversion_Weight']
    
    # Round to whole numbers (can't have fractional tickets)
    combined_df['Distributed_Single_Day'] = combined_df['Distributed_Single_Day'].round().astype(int)
    combined_df['Distributed_Annual_Passes'] = combined_df['Distributed_Annual_Passes'].round().astype(int)
    combined_df['Distributed_Reservations'] = combined_df['Distributed_Reservations'].round().astype(int)
    
    # Calculate total distributed tickets
    combined_df['Total_Distributed_Tickets'] = (combined_df['Distributed_Single_Day'] + 
                                               combined_df['Distributed_Annual_Passes'] + 
                                               combined_df['Distributed_Reservations'])
    
    print("Ticket distribution completed.")
    return combined_df

def add_performance_metrics(combined_df):
    """Add additional performance metrics to the combined dataset"""
    print("Adding performance metrics...")
    
    # Calculate efficiency metrics
    combined_df['Cost_Per_Ticket'] = np.where(combined_df['Total_Distributed_Tickets'] > 0,
                                             combined_df['Spend ($)'] / combined_df['Total_Distributed_Tickets'],
                                             0)
    
    combined_df['Ticket_Conversion_Rate'] = np.where(combined_df['Clicks'] > 0,
                                                    combined_df['Total_Distributed_Tickets'] / combined_df['Clicks'] * 100,
                                                    0)
    
    # Calculate ROAS based on distributed tickets
    combined_df['ROAS'] = np.where(combined_df['Spend ($)'] > 0,
                                  combined_df['Revenue ($)'] / combined_df['Spend ($)'],
                                  0)
    
    # Add ticket type percentages
    combined_df['Single_Day_Percentage'] = np.where(combined_df['Total_Distributed_Tickets'] > 0,
                                                   combined_df['Distributed_Single_Day'] / combined_df['Total_Distributed_Tickets'] * 100,
                                                   0)
    
    combined_df['Annual_Pass_Percentage'] = np.where(combined_df['Total_Distributed_Tickets'] > 0,
                                                    combined_df['Distributed_Annual_Passes'] / combined_df['Total_Distributed_Tickets'] * 100,
                                                    0)
    
    combined_df['Reservation_Percentage'] = np.where(combined_df['Total_Distributed_Tickets'] > 0,
                                                    combined_df['Distributed_Reservations'] / combined_df['Total_Distributed_Tickets'] * 100,
                                                    0)
    
    print("Performance metrics added.")
    return combined_df

def create_summary_reports(combined_df):
    """Create summary reports for analysis"""
    print("Creating summary reports...")
    
    # Channel performance summary
    channel_summary = combined_df.groupby(['Channel', 'Platform']).agg({
        'Spend ($)': 'sum',
        'Conversions': 'sum',
        'Revenue ($)': 'sum',
        'Distributed_Single_Day': 'sum',
        'Distributed_Annual_Passes': 'sum',
        'Distributed_Reservations': 'sum',
        'Total_Distributed_Tickets': 'sum'
    }).reset_index()
    
    channel_summary['Channel_ROAS'] = channel_summary['Revenue ($)'] / channel_summary['Spend ($)']
    channel_summary['Channel_Cost_Per_Ticket'] = channel_summary['Spend ($)'] / channel_summary['Total_Distributed_Tickets']
    
    # Segment performance summary
    segment_summary = combined_df.groupby(['Segment']).agg({
        'Spend ($)': 'sum',
        'Conversions': 'sum',
        'Revenue ($)': 'sum',
        'Distributed_Single_Day': 'sum',
        'Distributed_Annual_Passes': 'sum',
        'Distributed_Reservations': 'sum',
        'Total_Distributed_Tickets': 'sum'
    }).reset_index()
    
    segment_summary['Segment_ROAS'] = segment_summary['Revenue ($)'] / segment_summary['Spend ($)']
    segment_summary['Segment_Cost_Per_Ticket'] = segment_summary['Spend ($)'] / segment_summary['Total_Distributed_Tickets']
    
    # Park performance summary
    park_summary = combined_df.groupby(['Park']).agg({
        'Spend ($)': 'sum',
        'Conversions': 'sum',
        'Revenue ($)': 'sum',
        'Distributed_Single_Day': 'sum',
        'Distributed_Annual_Passes': 'sum',
        'Distributed_Reservations': 'sum',
        'Total_Distributed_Tickets': 'sum'
    }).reset_index()
    
    park_summary['Park_ROAS'] = park_summary['Revenue ($)'] / park_summary['Spend ($)']
    park_summary['Park_Cost_Per_Ticket'] = park_summary['Spend ($)'] / park_summary['Total_Distributed_Tickets']
    
    print("Summary reports created.")
    return channel_summary, segment_summary, park_summary

def save_results(combined_df, channel_summary, segment_summary, park_summary):
    """Save all results to CSV files"""
    print("Saving results...")
    
    # Create output directory if it doesn't exist
    output_dir = 'Combine Data/Output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main combined dataset
    combined_df.to_csv(f'{output_dir}/combined_paid_media_and_sales.csv', index=False)
    print(f"Combined dataset saved: {len(combined_df)} rows")
    
    # Save summary reports
    channel_summary.to_csv(f'{output_dir}/channel_performance_summary.csv', index=False)
    segment_summary.to_csv(f'{output_dir}/segment_performance_summary.csv', index=False)
    park_summary.to_csv(f'{output_dir}/park_performance_summary.csv', index=False)
    
    print("All files saved successfully!")

def validate_data_integrity(sales_df, combined_df):
    """Validate that ticket distributions match original totals"""
    print("\nValidating data integrity...")
    
    # Check if distributed tickets match original totals by week/park
    validation_check = combined_df.groupby(['Week', 'Park']).agg({
        'Distributed_Single_Day': 'sum',
        'Distributed_Annual_Passes': 'sum', 
        'Distributed_Reservations': 'sum'
    }).reset_index()
    
    validation_merged = validation_check.merge(sales_df[['Week', 'Park', 'Single-Day Tickets', 'Annual Passes', 'Reservations']], 
                                             on=['Week', 'Park'], how='left')
    
    # Calculate differences (should be minimal due to rounding)
    validation_merged['Single_Day_Diff'] = abs(validation_merged['Distributed_Single_Day'] - validation_merged['Single-Day Tickets'])
    validation_merged['Annual_Pass_Diff'] = abs(validation_merged['Distributed_Annual_Passes'] - validation_merged['Annual Passes'])
    validation_merged['Reservation_Diff'] = abs(validation_merged['Distributed_Reservations'] - validation_merged['Reservations'])
    
    max_single_diff = validation_merged['Single_Day_Diff'].max()
    max_annual_diff = validation_merged['Annual_Pass_Diff'].max()
    max_reservation_diff = validation_merged['Reservation_Diff'].max()
    
    print(f"Maximum differences (due to rounding):")
    print(f"Single-Day Tickets: {max_single_diff}")
    print(f"Annual Passes: {max_annual_diff}")
    print(f"Reservations: {max_reservation_diff}")
    
    if max_single_diff <= 5 and max_annual_diff <= 5 and max_reservation_diff <= 5:
        print("✅ Data integrity validation passed!")
    else:
        print("⚠️ Data integrity issues detected - please review")

def print_sample_results(combined_df):
    """Print sample results for review"""
    print("\nSample Results Preview:")
    print("="*80)
    
    # Show first few rows of key columns
    sample_cols = ['Week', 'Park', 'Segment', 'Channel', 'Platform', 'Conversions', 
                   'Distributed_Single_Day', 'Distributed_Annual_Passes', 'Distributed_Reservations',
                   'Cost_Per_Ticket', 'ROAS']
    
    print(combined_df[sample_cols].head(10).to_string(index=False))
    
    print("\nOverall Statistics:")
    print(f"Total Spend: ${combined_df['Spend ($)'].sum():,.2f}")
    print(f"Total Revenue: ${combined_df['Revenue ($)'].sum():,.2f}")
    print(f"Total Conversions: {combined_df['Conversions'].sum():,}")
    print(f"Total Distributed Single-Day Tickets: {combined_df['Distributed_Single_Day'].sum():,}")
    print(f"Total Distributed Annual Passes: {combined_df['Distributed_Annual_Passes'].sum():,}")
    print(f"Total Distributed Reservations: {combined_df['Distributed_Reservations'].sum():,}")
    print(f"Overall ROAS: {combined_df['Revenue ($)'].sum() / combined_df['Spend ($)'].sum():.2f}x")

def main():
    """Main execution function"""
    print("🎢 United Parks - Paid Media & Sales Data Combiner")
    print("="*60)
    
    # Load and clean data
    sales_df, media_df = load_and_clean_data()
    if sales_df is None or media_df is None:
        return
    
    # Calculate conversion weights
    media_weighted = calculate_conversion_weights(media_df)
    
    # Distribute ticket sales
    combined_df = distribute_ticket_sales(sales_df, media_weighted)
    
    # Add performance metrics
    combined_df = add_performance_metrics(combined_df)
    
    # Create summary reports
    channel_summary, segment_summary, park_summary = create_summary_reports(combined_df)
    
    # Validate data integrity
    validate_data_integrity(sales_df, combined_df)
    
    # Save results
    save_results(combined_df, channel_summary, segment_summary, park_summary)
    
    # Print sample results
    print_sample_results(combined_df)
    
    print("\n🎉 Data combination completed successfully!")
    print("Check the 'Combine Data/Output' folder for all generated files.")

if __name__ == "__main__":
    main() 