#!/usr/bin/env python3
"""
CSV to Excel Combiner
Combines multiple CSV files into a single Excel workbook with multiple sheets
Specifically designed for Busch Gardens Tampa Bay analysis outputs
"""

import pandas as pd
import os
import glob
from datetime import datetime

def combine_csvs_to_excel(park_prefix="united_parks_global", output_filename=None, source_directory="Global"):
    """
    Combine all CSV files with the specified park prefix into a single Excel file
    
    Args:
        park_prefix (str): Prefix of the CSV files to combine
        output_filename (str): Optional custom output filename
        source_directory (str): Directory containing the CSV files
    """
    
    # Get current working directory
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    print(f"Looking for CSV files in: {source_directory}")
    
    # Find all CSV files with the park prefix in the specified directory
    csv_pattern = os.path.join(source_directory, f"{park_prefix}_*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found with pattern: {csv_pattern}")
        return False
    
    print(f"Found {len(csv_files)} CSV files to combine:")
    for file in csv_files:
        print(f"  - {file}")
    
    # Create output filename if not provided
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{park_prefix}_combined_analysis_{timestamp}.xlsx"
    
    # Create Excel writer object
    try:
        with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
            
            for csv_file in sorted(csv_files):
                # Read CSV file
                print(f"Processing: {csv_file}")
                df = pd.read_csv(csv_file)
                
                # Create sheet name by removing directory path, park prefix and .csv extension
                base_filename = os.path.basename(csv_file)
                sheet_name = base_filename.replace(f"{park_prefix}_", "").replace(".csv", "")
                
                # Clean up sheet name for Excel (max 31 characters, no special chars)
                sheet_name = sheet_name.replace("_", " ").title()
                if len(sheet_name) > 31:
                    sheet_name = sheet_name[:31]
                
                # Write DataFrame to Excel sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50 characters
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                print(f"  ✓ Added sheet: '{sheet_name}' ({len(df)} rows, {len(df.columns)} columns)")
        
        print(f"\n✅ Successfully created Excel file: {output_filename}")
        
        # Display file size
        file_size = os.path.getsize(output_filename)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        print(f"File size: {size_str}")
        return True
        
    except ImportError:
        print("❌ Error: openpyxl library is required for Excel export.")
        print("Please install it using: pip install openpyxl")
        return False
    except Exception as e:
        print(f"❌ Error creating Excel file: {e}")
        return False

def create_summary_sheet(excel_filename, park_name="Busch Gardens Tampa Bay"):
    """
    Add a summary sheet to the Excel file with overview information
    """
    try:
        # Read the existing Excel file
        with pd.ExcelWriter(excel_filename, mode='a', engine='openpyxl') as writer:
            
            # Create summary data
            summary_data = {
                'Analysis Overview': [
                    f'Park: {park_name}',
                    f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                    f'Data Source: United Parks & Resorts Media Analysis',
                    '',
                    'Sheet Descriptions:',
                    '• Channel Performance: Performance metrics by marketing channel',
                    '• Segment Performance: Performance metrics by audience segment', 
                    '• Platform Performance: Performance metrics by advertising platform',
                    '• Optimization Opportunities: Budget reallocation recommendations',
                    '• Data Quality Issues: Summary of data accuracy problems identified',
                    '',
                    'Key Metrics Explained:',
                    '• ROAS: Return on Ad Spend (Revenue ÷ Spend)',
                    '• CPA: Cost Per Acquisition (Spend ÷ Conversions)',
                    '• CTR: Click-Through Rate (Clicks ÷ Impressions × 100)',
                    '• Revenue Share: Percentage of total revenue',
                    '• Spend Share: Percentage of total spend',
                    '',
                    'Notes:',
                    '• All financial figures are in USD',
                    '• Data has been cleaned to address quality issues',
                    '• Analysis covers full year of media performance',
                    '• Outliers have been statistically capped using IQR method'
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Format the summary sheet
            worksheet = writer.sheets['Summary']
            worksheet.column_dimensions['A'].width = 60
            
            print("✓ Added summary sheet")
            
    except Exception as e:
        print(f"Warning: Could not add summary sheet: {e}")

def main():
    """Main execution function"""
    print("🔄 CSV TO EXCEL COMBINER")
    print("=" * 50)
    
    # Check if openpyxl is available
    try:
        import openpyxl
        print("✓ openpyxl library found")
    except ImportError:
        print("❌ openpyxl library not found. Installing...")
        os.system("pip install openpyxl")
        try:
            import openpyxl
            print("✓ openpyxl library installed successfully")
        except ImportError:
            print("❌ Failed to install openpyxl. Please install manually:")
            print("   pip install openpyxl")
            return
    
    # Combine CSVs to Excel
    park_prefix = "busch_gardens_tampa_bay"
    source_directory = "Monthly"
    success = combine_csvs_to_excel(park_prefix=park_prefix, source_directory=source_directory)
    
    if success:
        # Find the created Excel file
        excel_files = glob.glob(f"{park_prefix}_combined_analysis_*.xlsx")
        if excel_files:
            latest_file = max(excel_files, key=os.path.getctime)
            print(f"\n📊 Adding summary sheet to: {latest_file}")
            create_summary_sheet(latest_file)
            
            print(f"\n🎉 COMPLETE! Your combined analysis is ready:")
            print(f"📁 File: {latest_file}")
            print(f"📊 Open this file in Excel to view all analysis sheets")
        else:
            print("❌ Could not find the created Excel file")
    else:
        print("❌ Failed to create Excel file")

if __name__ == "__main__":
    main() 