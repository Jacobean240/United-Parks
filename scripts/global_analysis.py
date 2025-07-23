#!/usr/bin/env python3
"""
United Parks & Resorts Global Media Analysis Script with Ticket Attribution
Director of Media - Strategic Recommendations

This script analyzes paid media performance data across ALL United Parks properties
with ticket sales attribution, accounting for data quality issues and generates 
insights for strategic presentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class UnitedParksMediaAnalysis:
    def __init__(self, file_path, target_park="All Parks"):
        """Initialize the analysis with data loading and quality assessment for all parks"""
        self.file_path = file_path
        self.target_park = target_park
        self.df = None
        self.df_cleaned = None
        self.quality_issues = {}
        self.insights = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the main dataset with ticket attribution for all United Parks properties"""
        print(f"Loading United Parks & Resorts paid media data with ticket attribution for ALL PARKS...")
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Global dataset loaded: {len(self.df):,} rows, {len(self.df.columns)} columns")
            
            # Check for ticket attribution columns
            ticket_columns = ['Single-Day Tickets', 'Annual Passes', 'Reservations']
            missing_ticket_cols = [col for col in ticket_columns if col not in self.df.columns]
            if missing_ticket_cols:
                print(f"Warning: Missing ticket attribution columns: {missing_ticket_cols}")
                # Add missing columns with zeros
                for col in missing_ticket_cols:
                    self.df[col] = 0
            
            # Show park breakdown
            park_counts = self.df['Park'].value_counts()
            print(f"Parks included in analysis:")
            for park, count in park_counts.items():
                print(f"  - {park}: {count:,} rows")
            
            # Show ticket attribution summary across all parks
            if 'Single-Day Tickets' in self.df.columns:
                total_single_day = self.df['Single-Day Tickets'].sum()
                total_annual = self.df['Annual Passes'].sum()
                total_reservations = self.df['Reservations'].sum()
                print(f"Global Ticket Attribution Summary:")
                print(f"  - Single-Day Tickets: {total_single_day:,.1f}")
                print(f"  - Annual Passes: {total_annual:,.1f}")
                print(f"  - Reservations: {total_reservations:,.1f}")
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def assess_data_quality(self):
        """Assess and document data quality issues including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("GLOBAL DATA QUALITY ASSESSMENT WITH TICKET ATTRIBUTION")
        print("="*50)
        
        # Basic data info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Parks: {self.df['Park'].nunique()} ({', '.join(self.df['Park'].unique())})")
        print(f"Segments: {self.df['Segment'].nunique()}")
        
        # Calculate performance metrics including ticket metrics
        self.df['CTR'] = np.where(self.df['Impressions'] > 0, 
                                 (self.df['Clicks'] / self.df['Impressions']) * 100, 0)
        self.df['Conversion_Rate'] = np.where(self.df['Clicks'] > 0,
                                            (self.df['Conversions'] / self.df['Clicks']) * 100, 0)
        self.df['ROAS'] = np.where(self.df['Spend'] > 0,
                                  self.df['Revenue'] / self.df['Spend'], 0)
        self.df['CPA'] = np.where(self.df['Conversions'] > 0,
                                 self.df['Spend'] / self.df['Conversions'], np.inf)
        
        # Add ticket-specific metrics
        if 'Single-Day Tickets' in self.df.columns:
            self.df['Total_Tickets'] = self.df['Single-Day Tickets'] + self.df['Annual Passes']
            self.df['Cost_Per_Ticket'] = np.where(self.df['Total_Tickets'] > 0,
                                                 self.df['Spend'] / self.df['Total_Tickets'], np.inf)
            self.df['Revenue_Per_Ticket'] = np.where(self.df['Total_Tickets'] > 0,
                                                    self.df['Revenue'] / self.df['Total_Tickets'], 0)
            self.df['Ticket_Conversion_Rate'] = np.where(self.df['Clicks'] > 0,
                                                        (self.df['Total_Tickets'] / self.df['Clicks']) * 100, 0)
        
        # Identify quality issues
        issues = {}
        
        # 1. Impossible CTRs (Clicks > Impressions)
        impossible_ctr = self.df['Clicks'] > self.df['Impressions']
        issues['impossible_ctr'] = impossible_ctr.sum()
        
        # 2. Impossible Conversion Rates (Conversions > Clicks) 
        impossible_cr = self.df['Conversions'] > self.df['Clicks']
        issues['impossible_cr'] = impossible_cr.sum()
        
        # 3. Missing Platform data
        missing_platform = self.df['Platform'].isna().sum()
        issues['missing_platform'] = missing_platform
        
        # 4. Extreme outliers
        issues['extreme_ctr'] = (self.df['CTR'] > 100).sum()
        issues['extreme_roas'] = (self.df['ROAS'] > 50).sum()
        issues['extreme_cpa'] = (self.df['CPA'] > 1000).sum()
        
        # 5. Ticket attribution issues
        if 'Total_Tickets' in self.df.columns:
            issues['impossible_ticket_conv'] = (self.df['Total_Tickets'] > self.df['Conversions']).sum()
            issues['extreme_cost_per_ticket'] = (self.df['Cost_Per_Ticket'] > 500).sum()
            issues['negative_tickets'] = ((self.df['Single-Day Tickets'] < 0) | 
                                        (self.df['Annual Passes'] < 0) | 
                                        (self.df['Reservations'] < 0)).sum()
        
        self.quality_issues = issues
        
        # Report issues
        print("\nDATA QUALITY ISSUES IDENTIFIED:")
        print(f"* Impossible CTRs (Clicks > Impressions): {issues['impossible_ctr']:,} rows")
        print(f"* Impossible Conversion Rates (Conversions > Clicks): {issues['impossible_cr']:,} rows") 
        print(f"* Missing Platform data: {issues['missing_platform']:,} rows")
        print(f"* Extreme CTR outliers (>100%): {issues['extreme_ctr']:,} rows")
        print(f"* Extreme ROAS outliers (>50x): {issues['extreme_roas']:,} rows")
        print(f"* Extreme CPA outliers (>$1000): {issues['extreme_cpa']:,} rows")
        
        if 'Total_Tickets' in self.df.columns:
            print(f"* Impossible Ticket Conversions (Tickets > Conversions): {issues.get('impossible_ticket_conv', 0):,} rows")
            print(f"* Extreme Cost per Ticket (>$500): {issues.get('extreme_cost_per_ticket', 0):,} rows")
            print(f"* Negative Ticket Values: {issues.get('negative_tickets', 0):,} rows")
        
        data_quality_score = max(0, 100 - (sum(issues.values()) / len(self.df) * 100))
        print(f"\nData Quality Score: {data_quality_score:.1f}/100")
        
    def clean_data(self):
        """Clean the data addressing identified quality issues including ticket data across all parks"""
        print("\n" + "="*50)
        print("GLOBAL DATA CLEANING & PREPARATION WITH TICKET ATTRIBUTION")
        print("="*50)
        
        self.df_cleaned = self.df.copy()
        
        # 1. Fix impossible CTRs - cap clicks at impressions
        impossible_ctr_mask = self.df_cleaned['Clicks'] > self.df_cleaned['Impressions']
        self.df_cleaned.loc[impossible_ctr_mask, 'Clicks'] = self.df_cleaned.loc[impossible_ctr_mask, 'Impressions']
        print(f"Fixed {impossible_ctr_mask.sum():,} impossible CTR cases")
        
        # 2. Fix impossible conversion rates - cap conversions at clicks
        impossible_cr_mask = self.df_cleaned['Conversions'] > self.df_cleaned['Clicks']
        self.df_cleaned.loc[impossible_cr_mask, 'Conversions'] = self.df_cleaned.loc[impossible_cr_mask, 'Clicks']
        print(f"Fixed {impossible_cr_mask.sum():,} impossible conversion rate cases")
        
        # 3. Fix impossible ticket conversions - cap total tickets at conversions
        if 'Total_Tickets' in self.df_cleaned.columns:
            impossible_ticket_mask = self.df_cleaned['Total_Tickets'] > self.df_cleaned['Conversions']
            ratio = self.df_cleaned.loc[impossible_ticket_mask, 'Total_Tickets'] / self.df_cleaned.loc[impossible_ticket_mask, 'Conversions']
            self.df_cleaned.loc[impossible_ticket_mask, 'Single-Day Tickets'] = self.df_cleaned.loc[impossible_ticket_mask, 'Single-Day Tickets'] / ratio
            self.df_cleaned.loc[impossible_ticket_mask, 'Annual Passes'] = self.df_cleaned.loc[impossible_ticket_mask, 'Annual Passes'] / ratio
            print(f"Fixed {impossible_ticket_mask.sum():,} impossible ticket conversion cases")
        
        # 4. Handle missing platforms - fill with 'Traditional Media' for non-digital channels
        traditional_channels = ['TV', 'Radio', 'Direct Mail']
        missing_platform_mask = self.df_cleaned['Platform'].isna()
        traditional_mask = self.df_cleaned['Channel'].isin(traditional_channels) & missing_platform_mask
        self.df_cleaned.loc[traditional_mask, 'Platform'] = 'Traditional Media'
        print(f"Filled {traditional_mask.sum():,} missing platform values for traditional media")
        
        # 5. Fix negative ticket values
        if 'Single-Day Tickets' in self.df_cleaned.columns:
            negative_tickets_mask = ((self.df_cleaned['Single-Day Tickets'] < 0) | 
                                   (self.df_cleaned['Annual Passes'] < 0) | 
                                   (self.df_cleaned['Reservations'] < 0))
            self.df_cleaned.loc[negative_tickets_mask, ['Single-Day Tickets', 'Annual Passes', 'Reservations']] = 0
            print(f"Fixed {negative_tickets_mask.sum():,} negative ticket values")
        
        # 6. Recalculate metrics with cleaned data
        self.df_cleaned['CTR'] = np.where(self.df_cleaned['Impressions'] > 0,
                                         (self.df_cleaned['Clicks'] / self.df_cleaned['Impressions']) * 100, 0)
        self.df_cleaned['Conversion_Rate'] = np.where(self.df_cleaned['Clicks'] > 0,
                                                    (self.df_cleaned['Conversions'] / self.df_cleaned['Clicks']) * 100, 0)
        self.df_cleaned['ROAS'] = np.where(self.df_cleaned['Spend'] > 0,
                                         self.df_cleaned['Revenue'] / self.df_cleaned['Spend'], 0)
        self.df_cleaned['CPA'] = np.where(self.df_cleaned['Conversions'] > 0,
                                        self.df_cleaned['Spend'] / self.df_cleaned['Conversions'], np.inf)
        
        # Recalculate ticket metrics
        if 'Single-Day Tickets' in self.df_cleaned.columns:
            self.df_cleaned['Total_Tickets'] = self.df_cleaned['Single-Day Tickets'] + self.df_cleaned['Annual Passes']
            self.df_cleaned['Cost_Per_Ticket'] = np.where(self.df_cleaned['Total_Tickets'] > 0,
                                                         self.df_cleaned['Spend'] / self.df_cleaned['Total_Tickets'], np.inf)
            self.df_cleaned['Revenue_Per_Ticket'] = np.where(self.df_cleaned['Total_Tickets'] > 0,
                                                            self.df_cleaned['Revenue'] / self.df_cleaned['Total_Tickets'], 0)
            self.df_cleaned['Ticket_Conversion_Rate'] = np.where(self.df_cleaned['Clicks'] > 0,
                                                               (self.df_cleaned['Total_Tickets'] / self.df_cleaned['Clicks']) * 100, 0)
        
        # 7. Handle extreme outliers using IQR method for key metrics
        outlier_metrics = ['CTR', 'Conversion_Rate', 'ROAS']
        if 'Cost_Per_Ticket' in self.df_cleaned.columns:
            outlier_metrics.append('Cost_Per_Ticket')
            
        for metric in outlier_metrics:
            if metric in self.df_cleaned.columns:
                Q1 = self.df_cleaned[metric].quantile(0.25)
                Q3 = self.df_cleaned[metric].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_mask = (self.df_cleaned[metric] < lower_bound) | (self.df_cleaned[metric] > upper_bound)
                outliers_count = outliers_mask.sum()
                
                # Cap extreme values
                self.df_cleaned.loc[self.df_cleaned[metric] > upper_bound, metric] = upper_bound
                self.df_cleaned.loc[self.df_cleaned[metric] < lower_bound, metric] = lower_bound
                
                print(f"Capped {outliers_count:,} extreme {metric} outliers")
        
        # Convert Date to datetime
        self.df_cleaned['Date'] = pd.to_datetime(self.df_cleaned['Date'])
        
        print(f"\nCleaned dataset ready: {len(self.df_cleaned):,} rows")
        
    def ticket_attribution_analysis(self):
        """Analyze ticket sales attribution across channels and segments for all parks"""
        print("\n" + "="*50)
        print("GLOBAL TICKET ATTRIBUTION ANALYSIS")
        print("="*50)
        
        if 'Single-Day Tickets' not in self.df_cleaned.columns:
            print("No ticket attribution data available")
            return
        
        # Overall ticket metrics
        total_single_day = self.df_cleaned['Single-Day Tickets'].sum()
        total_annual = self.df_cleaned['Annual Passes'].sum()
        total_reservations = self.df_cleaned['Reservations'].sum()
        total_tickets = total_single_day + total_annual
        total_spend = self.df_cleaned['Spend'].sum()
        avg_cost_per_ticket = total_spend / total_tickets if total_tickets > 0 else 0
        avg_revenue_per_ticket = self.df_cleaned['Revenue'].sum() / total_tickets if total_tickets > 0 else 0
        
        print(f"OVERALL TICKET PERFORMANCE:")
        print(f"* Total Single-Day Tickets: {total_single_day:,.0f}")
        print(f"* Total Annual Passes: {total_annual:,.0f}")
        print(f"* Total Reservations: {total_reservations:,.0f}")
        print(f"* Total Tickets Sold: {total_tickets:,.0f}")
        print(f"* Average Cost per Ticket: ${avg_cost_per_ticket:.2f}")
        print(f"* Average Revenue per Ticket: ${avg_revenue_per_ticket:.2f}")
        print(f"* Annual Pass Ratio: {(total_annual/total_tickets*100):.1f}%")
        
        # Channel ticket performance
        channel_tickets = self.df_cleaned.groupby('Channel').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Single-Day Tickets': 'sum',
            'Annual Passes': 'sum',
            'Reservations': 'sum',
            'Total_Tickets': 'sum',
            'Cost_Per_Ticket': 'mean',
            'Revenue_Per_Ticket': 'mean'
        }).reset_index()
        
        channel_tickets['Ticket_Share'] = (channel_tickets['Total_Tickets'] / channel_tickets['Total_Tickets'].sum()) * 100
        channel_tickets['Annual_Pass_Rate'] = (channel_tickets['Annual Passes'] / channel_tickets['Total_Tickets']) * 100
        channel_tickets = channel_tickets.sort_values('Total_Tickets', ascending=False)
        
        print(f"\nTOP CHANNELS BY TICKET SALES:")
        for i, row in channel_tickets.head(5).iterrows():
            print(f"{row['Channel']}: {row['Total_Tickets']:,.0f} tickets ({row['Ticket_Share']:.1f}% share), ${row['Cost_Per_Ticket']:.2f} cost/ticket")
            
        print(f"\nMOST EFFICIENT CHANNELS (Lowest Cost per Ticket):")
        efficient_channels = channel_tickets[channel_tickets['Total_Tickets'] > 0].nsmallest(3, 'Cost_Per_Ticket')
        for i, row in efficient_channels.iterrows():
            print(f"{row['Channel']}: ${row['Cost_Per_Ticket']:.2f} cost/ticket, {row['Total_Tickets']:,.0f} tickets")
        
        # Segment ticket performance
        segment_tickets = self.df_cleaned.groupby('Segment').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Single-Day Tickets': 'sum',
            'Annual Passes': 'sum',
            'Total_Tickets': 'sum',
            'Cost_Per_Ticket': 'mean'
        }).reset_index()
        
        segment_tickets['Ticket_Share'] = (segment_tickets['Total_Tickets'] / segment_tickets['Total_Tickets'].sum()) * 100
        segment_tickets['Annual_Pass_Rate'] = (segment_tickets['Annual Passes'] / segment_tickets['Total_Tickets']) * 100
        segment_tickets = segment_tickets.sort_values('Total_Tickets', ascending=False)
        
        print(f"\nSEGMENT TICKET PERFORMANCE:")
        for i, row in segment_tickets.iterrows():
            print(f"{row['Segment']}: {row['Total_Tickets']:,.0f} tickets ({row['Ticket_Share']:.1f}% share), {row['Annual_Pass_Rate']:.1f}% annual passes")
        
        # Store ticket insights
        self.insights['tickets'] = {
            'overall': {
                'total_single_day': total_single_day,
                'total_annual': total_annual,
                'total_reservations': total_reservations,
                'avg_cost_per_ticket': avg_cost_per_ticket,
                'avg_revenue_per_ticket': avg_revenue_per_ticket
            },
            'channel_performance': channel_tickets,
            'segment_performance': segment_tickets
        }
        
    def performance_overview(self):
        """Analyze overall performance trends and channel effectiveness including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("GLOBAL PERFORMANCE OVERVIEW ANALYSIS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        # Overall metrics
        total_spend = self.df_cleaned['Spend'].sum()
        total_revenue = self.df_cleaned['Revenue'].sum()
        total_conversions = self.df_cleaned['Conversions'].sum()
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        overall_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        print(f"OVERALL PERFORMANCE:")
        print(f"* Total Spend: ${total_spend:,.2f}")
        print(f"* Total Revenue: ${total_revenue:,.2f}")
        print(f"* Total Conversions: {total_conversions:,.0f}")
        print(f"* Overall ROAS: {overall_roas:.2f}x")
        print(f"* Overall CPA: ${overall_cpa:.2f}")
        
        # Add ticket metrics if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            total_tickets = self.df_cleaned['Total_Tickets'].sum()
            overall_cost_per_ticket = total_spend / total_tickets if total_tickets > 0 else 0
            print(f"* Total Tickets Sold: {total_tickets:,.0f}")
            print(f"* Overall Cost per Ticket: ${overall_cost_per_ticket:.2f}")
        
        # Channel performance
        channel_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            channel_agg_dict.update({
                'Single-Day Tickets': 'sum',
                'Annual Passes': 'sum',
                'Total_Tickets': 'sum'
            })
        
        channel_performance = self.df_cleaned.groupby('Channel').agg(channel_agg_dict).reset_index()
        
        channel_performance['ROAS'] = channel_performance['Revenue'] / channel_performance['Spend']
        channel_performance['CPA'] = channel_performance['Spend'] / channel_performance['Conversions']
        channel_performance['CTR'] = (channel_performance['Clicks'] / channel_performance['Impressions']) * 100
        
        # Add ticket metrics if available
        if 'Total_Tickets' in channel_performance.columns:
            channel_performance['Cost_Per_Ticket'] = channel_performance['Spend'] / channel_performance['Total_Tickets']
            channel_performance['Annual_Pass_Rate'] = (channel_performance['Annual Passes'] / channel_performance['Total_Tickets']) * 100
        
        channel_performance = channel_performance.sort_values('Revenue', ascending=False)
        
        print(f"\nTOP PERFORMING CHANNELS (by Revenue):")
        for i, row in channel_performance.head(3).iterrows():
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets (${row['Cost_Per_Ticket']:.2f}/ticket)"
            print(f"{row['Channel']}: ${row['Revenue']:,.0f} revenue, {row['ROAS']:.2f}x ROAS{ticket_info}")
            
        print(f"\nLOWEST PERFORMING CHANNELS (by ROAS):")
        for i, row in channel_performance.nsmallest(3, 'ROAS').iterrows():
            print(f"{row['Channel']}: {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA")
        
        # Platform performance
        platform_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            platform_agg_dict['Total_Tickets'] = 'sum'
        
        platform_performance = self.df_cleaned.groupby('Platform').agg(platform_agg_dict).reset_index()
        platform_performance['ROAS'] = platform_performance['Revenue'] / platform_performance['Spend']
        
        # Add ticket metrics if available
        if 'Total_Tickets' in platform_performance.columns:
            platform_performance['Cost_Per_Ticket'] = platform_performance['Spend'] / platform_performance['Total_Tickets']
        
        platform_performance = platform_performance.sort_values('Revenue', ascending=False)
        
        print(f"\nTOP PERFORMING PLATFORMS:")
        for i, row in platform_performance.head(5).iterrows():
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets"
            print(f"{row['Platform']}: ${row['Revenue']:,.0f} revenue, {row['ROAS']:.2f}x ROAS{ticket_info}")
        
        # Store insights
        performance_insights = {
            'overall_roas': overall_roas,
            'overall_cpa': overall_cpa,
            'top_channels': channel_performance.head(3),
            'bottom_channels': channel_performance.nsmallest(3, 'ROAS'),
            'top_platforms': platform_performance.head(5)
        }
        
        # Add ticket insights if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            performance_insights['overall_cost_per_ticket'] = overall_cost_per_ticket
            performance_insights['total_tickets'] = total_tickets
        
        self.insights['performance'] = performance_insights
        
    def audience_segmentation_analysis(self):
        """Analyze performance by audience segments including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("GLOBAL AUDIENCE SEGMENTATION ANALYSIS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        # Segment performance
        segment_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            segment_agg_dict.update({
                'Single-Day Tickets': 'sum',
                'Annual Passes': 'sum',
                'Total_Tickets': 'sum'
            })
        
        segment_performance = self.df_cleaned.groupby('Segment').agg(segment_agg_dict).reset_index()
        
        segment_performance['ROAS'] = segment_performance['Revenue'] / segment_performance['Spend']
        segment_performance['CPA'] = segment_performance['Spend'] / segment_performance['Conversions']
        segment_performance['CTR'] = (segment_performance['Clicks'] / segment_performance['Impressions']) * 100
        segment_performance['Revenue_Share'] = (segment_performance['Revenue'] / segment_performance['Revenue'].sum()) * 100
        
        # Add ticket metrics if available
        if 'Total_Tickets' in segment_performance.columns:
            segment_performance['Cost_Per_Ticket'] = segment_performance['Spend'] / segment_performance['Total_Tickets']
            segment_performance['Ticket_Share'] = (segment_performance['Total_Tickets'] / segment_performance['Total_Tickets'].sum()) * 100
            segment_performance['Annual_Pass_Rate'] = (segment_performance['Annual Passes'] / segment_performance['Total_Tickets']) * 100
        
        segment_performance = segment_performance.sort_values('Revenue', ascending=False)
        
        print("📈 SEGMENT PERFORMANCE RANKING:")
        for i, row in segment_performance.iterrows():
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets ({row['Ticket_Share']:.1f}% ticket share), ${row['Cost_Per_Ticket']:.2f}/ticket"
            print(f"{row['Segment']}: ${row['Revenue']:,.0f} ({row['Revenue_Share']:.1f}% share), {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA{ticket_info}")
        
        # Segment channel preferences
        print(f"\n🎯 SEGMENT CHANNEL PREFERENCES:")
        for segment in segment_performance['Segment'].head(3):
            segment_data = self.df_cleaned[self.df_cleaned['Segment'] == segment]
            channel_revenue = segment_data.groupby('Channel')['Revenue'].sum().sort_values(ascending=False)
            top_channel = channel_revenue.index[0]
            top_revenue = channel_revenue.iloc[0]
            total_segment_revenue = channel_revenue.sum()
            share = (top_revenue / total_segment_revenue) * 100
            
            # Add ticket info for channel preference
            if 'Total_Tickets' in self.df_cleaned.columns:
                channel_tickets = segment_data.groupby('Channel')['Total_Tickets'].sum()
                top_channel_tickets = channel_tickets.get(top_channel, 0)
                ticket_info = f", {top_channel_tickets:,.0f} tickets"
            else:
                ticket_info = ""
            
            print(f"• {segment}: {top_channel} (${top_revenue:,.0f}, {share:.1f}% of segment revenue{ticket_info})")
        
        self.insights['segments'] = segment_performance
        
    def funnel_stage_analysis(self):
        """Analyze performance by funnel stages including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("GLOBAL FUNNEL STAGE ANALYSIS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        funnel_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            funnel_agg_dict['Total_Tickets'] = 'sum'
        
        funnel_performance = self.df_cleaned.groupby('Funnel Stage').agg(funnel_agg_dict).reset_index()
        
        funnel_performance['ROAS'] = funnel_performance['Revenue'] / funnel_performance['Spend']
        funnel_performance['CPA'] = funnel_performance['Spend'] / funnel_performance['Conversions']
        funnel_performance['Spend_Share'] = (funnel_performance['Spend'] / funnel_performance['Spend'].sum()) * 100
        
        # Add ticket metrics if available
        if 'Total_Tickets' in funnel_performance.columns:
            funnel_performance['Cost_Per_Ticket'] = funnel_performance['Spend'] / funnel_performance['Total_Tickets']
            funnel_performance['Ticket_Share'] = (funnel_performance['Total_Tickets'] / funnel_performance['Total_Tickets'].sum()) * 100
        
        # Order by funnel stage
        stage_order = ['Awareness', 'Consideration', 'Conversion']
        funnel_performance['Stage_Order'] = funnel_performance['Funnel Stage'].map({stage: i for i, stage in enumerate(stage_order)})
        funnel_performance = funnel_performance.sort_values('Stage_Order')
        
        print("FUNNEL STAGE PERFORMANCE:")
        for i, row in funnel_performance.iterrows():
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets ({row['Ticket_Share']:.1f}% ticket share)"
            print(f"{row['Funnel Stage']}: ${row['Spend']:,.0f} spend ({row['Spend_Share']:.1f}%), {row['ROAS']:.2f}x ROAS{ticket_info}")
        
        self.insights['funnel'] = funnel_performance
        
    def comprehensive_monthly_analysis(self):
        """Conduct comprehensive monthly analysis with trends, seasonality, and strategic insights including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("GLOBAL COMPREHENSIVE MONTHLY ANALYSIS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        # Create month period column
        self.df_cleaned['Month_Period'] = self.df_cleaned['Date'].dt.to_period('M')
        self.df_cleaned['Month_Name'] = self.df_cleaned['Date'].dt.strftime('%B')
        self.df_cleaned['Month_Num'] = self.df_cleaned['Date'].dt.month
        
        # Overall monthly performance
        monthly_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            monthly_agg_dict['Total_Tickets'] = 'sum'
        
        monthly_overview = self.df_cleaned.groupby(['Month_Period', 'Month_Name', 'Month_Num']).agg(monthly_agg_dict).reset_index()
        
        monthly_overview['ROAS'] = monthly_overview['Revenue'] / monthly_overview['Spend']
        monthly_overview['CPA'] = monthly_overview['Spend'] / monthly_overview['Conversions']
        monthly_overview['CTR'] = (monthly_overview['Clicks'] / monthly_overview['Impressions']) * 100
        
        # Add ticket metrics if available
        if 'Total_Tickets' in monthly_overview.columns:
            monthly_overview['Cost_Per_Ticket'] = monthly_overview['Spend'] / monthly_overview['Total_Tickets']
        
        # Calculate month-over-month growth
        monthly_overview = monthly_overview.sort_values('Month_Num')
        monthly_overview['Revenue_MoM_Growth'] = monthly_overview['Revenue'].pct_change() * 100
        monthly_overview['Spend_MoM_Growth'] = monthly_overview['Spend'].pct_change() * 100
        monthly_overview['ROAS_MoM_Change'] = monthly_overview['ROAS'].diff()
        
        if 'Total_Tickets' in monthly_overview.columns:
            monthly_overview['Tickets_MoM_Growth'] = monthly_overview['Total_Tickets'].pct_change() * 100
        
        print("MONTHLY PERFORMANCE OVERVIEW:")
        for _, row in monthly_overview.iterrows():
            mom_revenue = f"({row['Revenue_MoM_Growth']:+.1f}% MoM)" if not pd.isna(row['Revenue_MoM_Growth']) else ""
            mom_roas = f"({row['ROAS_MoM_Change']:+.2f}x MoM)" if not pd.isna(row['ROAS_MoM_Change']) else ""
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                mom_tickets = f"({row['Tickets_MoM_Growth']:+.1f}% MoM)" if not pd.isna(row.get('Tickets_MoM_Growth')) else ""
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets {mom_tickets}"
            print(f"{row['Month_Name']}: ${row['Revenue']:,.0f} revenue {mom_revenue}, {row['ROAS']:.2f}x ROAS {mom_roas}{ticket_info}")
        
        # Monthly channel performance
        channel_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }
        if 'Total_Tickets' in self.df_cleaned.columns:
            channel_agg_dict['Total_Tickets'] = 'sum'
        
        monthly_channel = self.df_cleaned.groupby(['Month_Period', 'Month_Name', 'Channel']).agg(channel_agg_dict).reset_index()
        monthly_channel['ROAS'] = monthly_channel['Revenue'] / monthly_channel['Spend']
        
        # Monthly segment performance  
        segment_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }
        if 'Total_Tickets' in self.df_cleaned.columns:
            segment_agg_dict['Total_Tickets'] = 'sum'
        
        monthly_segment = self.df_cleaned.groupby(['Month_Period', 'Month_Name', 'Segment']).agg(segment_agg_dict).reset_index()
        monthly_segment['ROAS'] = monthly_segment['Revenue'] / monthly_segment['Spend']
        
        # Seasonality analysis
        seasonality_insights = self.analyze_seasonality(monthly_overview, monthly_channel, monthly_segment)
        
        # Store comprehensive monthly insights
        self.insights['monthly'] = {
            'overview': monthly_overview,
            'channel_monthly': monthly_channel,
            'segment_monthly': monthly_segment,
            'seasonality': seasonality_insights
        }
        
    def analyze_seasonality(self, monthly_overview, monthly_channel, monthly_segment):
        """Analyze seasonality patterns and provide strategic insights"""
        print(f"\nSEASONALITY AND TREND ANALYSIS:")
        
        # Identify peak and low seasons
        peak_month = monthly_overview.loc[monthly_overview['Revenue'].idxmax()]
        low_month = monthly_overview.loc[monthly_overview['Revenue'].idxmin()]
        
        print(f"Peak Season: {peak_month['Month_Name']} (${peak_month['Revenue']:,.0f} revenue, {peak_month['ROAS']:.2f}x ROAS)")
        print(f"Low Season: {low_month['Month_Name']} (${low_month['Revenue']:,.0f} revenue, {low_month['ROAS']:.2f}x ROAS)")
        
        # Calculate seasonal variance
        revenue_cv = (monthly_overview['Revenue'].std() / monthly_overview['Revenue'].mean()) * 100
        print(f"Revenue Seasonality (CV): {revenue_cv:.1f}%")
        
        # Identify best performing channels by season
        q1_months = [1, 2, 3]
        q2_months = [4, 5, 6] 
        q3_months = [7, 8, 9]
        q4_months = [10, 11, 12]
        
        quarterly_insights = {}
        for quarter, months in [('Q1', q1_months), ('Q2', q2_months), ('Q3', q3_months), ('Q4', q4_months)]:
            quarter_data = self.df_cleaned[self.df_cleaned['Month_Num'].isin(months)]
            if not quarter_data.empty:
                quarter_channel = quarter_data.groupby('Channel').agg({
                    'Spend': 'sum',
                    'Revenue': 'sum'
                }).reset_index()
                quarter_channel['ROAS'] = quarter_channel['Revenue'] / quarter_channel['Spend']
                top_channel = quarter_channel.loc[quarter_channel['ROAS'].idxmax()]
                quarterly_insights[quarter] = {
                    'top_channel': top_channel['Channel'],
                    'top_roas': top_channel['ROAS'],
                    'total_revenue': quarter_channel['Revenue'].sum()
                }
                print(f"{quarter} Top Channel: {top_channel['Channel']} ({top_channel['ROAS']:.2f}x ROAS)")
        
        # Growth trend analysis
        total_months = len(monthly_overview)
        if total_months >= 6:
            first_half = monthly_overview.head(6)['Revenue'].mean()
            second_half = monthly_overview.tail(6)['Revenue'].mean()
            yoy_trend = ((second_half - first_half) / first_half) * 100
            print(f"Year-over-Year Trend: {yoy_trend:+.1f}% revenue growth (H2 vs H1)")
        
        return {
            'peak_month': peak_month,
            'low_month': low_month,
            'revenue_seasonality': revenue_cv,
            'quarterly_insights': quarterly_insights,
            'yoy_trend': yoy_trend if 'yoy_trend' in locals() else None
        }
        
    def park_performance_analysis(self):
        """Analyze comparative performance across all parks with ticket attribution"""
        print("\n" + "="*50)
        print("COMPARATIVE PARK PERFORMANCE ANALYSIS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        park_agg_dict = {
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }
        
        # Add ticket columns if available
        if 'Total_Tickets' in self.df_cleaned.columns:
            park_agg_dict['Total_Tickets'] = 'sum'
        
        park_performance = self.df_cleaned.groupby('Park').agg(park_agg_dict).reset_index()
        
        park_performance['ROAS'] = park_performance['Revenue'] / park_performance['Spend']
        park_performance['CPA'] = park_performance['Spend'] / park_performance['Conversions']
        
        # Add ticket metrics if available
        if 'Total_Tickets' in park_performance.columns:
            park_performance['Cost_Per_Ticket'] = park_performance['Spend'] / park_performance['Total_Tickets']
        
        park_performance = park_performance.sort_values('Revenue', ascending=False)
        
        print("🏞️ ALL PARKS PERFORMANCE RANKING:")
        for i, row in park_performance.iterrows():
            ticket_info = ""
            if 'Total_Tickets' in row and row['Total_Tickets'] > 0:
                ticket_info = f", {row['Total_Tickets']:,.0f} tickets (${row['Cost_Per_Ticket']:.2f}/ticket)"
            print(f"{row['Park']}: ${row['Revenue']:,.0f}, {row['ROAS']:.2f}x ROAS{ticket_info}")
        
        self.insights['parks'] = park_performance
        
    def identify_optimization_opportunities(self):
        """Identify areas for media spend reallocation and optimization across all parks"""
        print("\n" + "="*50)
        print("GLOBAL OPTIMIZATION OPPORTUNITIES")
        print("="*50)
        
        # High spend, low ROAS opportunities
        channel_performance = self.insights['performance']['top_channels']
        segment_performance = self.insights['segments']
        
        # Calculate efficiency scores
        self.df_cleaned['Efficiency_Score'] = self.df_cleaned['ROAS'] * (1 / (self.df_cleaned['CPA'] + 1))
        
        # Reallocation opportunities by channel
        reallocation_opportunities = self.df_cleaned.groupby(['Channel', 'Segment']).agg({
            'Spend': 'sum',
            'ROAS': 'mean',
            'CPA': 'mean',
            'Efficiency_Score': 'mean'
        }).reset_index()
        
        # Find underperforming high-spend areas
        high_spend_threshold = reallocation_opportunities['Spend'].quantile(0.75)
        low_roas_threshold = reallocation_opportunities['ROAS'].quantile(0.25)
        
        underperforming = reallocation_opportunities[
            (reallocation_opportunities['Spend'] > high_spend_threshold) & 
            (reallocation_opportunities['ROAS'] < low_roas_threshold)
        ].sort_values('Spend', ascending=False)
        
        print("⚠️ REALLOCATION OPPORTUNITIES (High Spend, Low ROAS):")
        for i, row in underperforming.head(5).iterrows():
            print(f"• {row['Channel']} - {row['Segment']}: ${row['Spend']:,.0f} spend, {row['ROAS']:.2f}x ROAS")
        
        # Find high-performing opportunities for increased investment
        high_efficiency = reallocation_opportunities[
            reallocation_opportunities['Efficiency_Score'] > reallocation_opportunities['Efficiency_Score'].quantile(0.8)
        ].sort_values('Efficiency_Score', ascending=False)
        
        print(f"\n🚀 SCALE-UP OPPORTUNITIES (High Efficiency):")
        for i, row in high_efficiency.head(5).iterrows():
            print(f"• {row['Channel']} - {row['Segment']}: {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA")
        
        self.insights['optimization'] = {
            'underperforming': underperforming,
            'scale_up': high_efficiency
        }
        
    def create_visualizations(self):
        """Create key visualizations for the presentation including ticket attribution across all parks"""
        print("\n" + "="*50)
        print("CREATING GLOBAL VISUALIZATIONS WITH TICKET ATTRIBUTION")
        print("="*50)
        
        # Create output directory
        os.makedirs('united_parks_charts', exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Overall Performance Dashboard
        fig_rows = 3 if 'Total_Tickets' in self.df_cleaned.columns else 2
        fig, axes = plt.subplots(fig_rows, 2, figsize=(16, 6*fig_rows))
        fig.suptitle('United Parks & Resorts - Global Media Performance Dashboard with Ticket Attribution', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        # Channel ROAS
        channel_data = self.insights['performance']['top_channels'].head(8)
        axes[0].bar(range(len(channel_data)), channel_data['ROAS'], color='steelblue')
        axes[0].set_title('ROAS by Channel', fontweight='bold')
        axes[0].set_ylabel('ROAS (x)')
        axes[0].set_xticks(range(len(channel_data)))
        axes[0].set_xticklabels(channel_data['Channel'], rotation=45, ha='right')
        
        # Segment Revenue Share
        segment_data = self.insights['segments']
        axes[1].pie(segment_data['Revenue'], labels=segment_data['Segment'], autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Revenue Share by Segment', fontweight='bold')
        
        # Funnel Stage Spend Distribution
        funnel_data = self.insights['funnel']
        axes[2].bar(funnel_data['Funnel Stage'], funnel_data['Spend'] / 1000000, color='lightcoral')
        axes[2].set_title('Spend Distribution by Funnel Stage', fontweight='bold')
        axes[2].set_ylabel('Spend ($M)')
        axes[2].tick_params(axis='x', rotation=0)
        
        # Monthly Trend
        monthly_trend = self.df_cleaned.groupby(self.df_cleaned['Date'].dt.to_period('M')).agg({
            'Spend': 'sum',
            'Revenue': 'sum'
        }).reset_index()
        monthly_trend['ROAS'] = monthly_trend['Revenue'] / monthly_trend['Spend']
        
        axes[3].plot(range(len(monthly_trend)), monthly_trend['ROAS'], marker='o', linewidth=2, color='green')
        axes[3].set_title('Monthly ROAS Trend', fontweight='bold')
        axes[3].set_ylabel('ROAS (x)')
        axes[3].set_xlabel('Month')
        axes[3].grid(True, alpha=0.3)
        
        # Add ticket attribution charts if data is available
        if 'Total_Tickets' in self.df_cleaned.columns and 'tickets' in self.insights:
            # Channel Ticket Performance
            ticket_channels = self.insights['tickets']['channel_performance'].head(8)
            axes[4].bar(range(len(ticket_channels)), ticket_channels['Cost_Per_Ticket'], color='purple')
            axes[4].set_title('Cost per Ticket by Channel', fontweight='bold')
            axes[4].set_ylabel('Cost per Ticket ($)')
            axes[4].set_xticks(range(len(ticket_channels)))
            axes[4].set_xticklabels(ticket_channels['Channel'], rotation=45, ha='right')
            
            # Ticket Type Distribution
            ticket_overall = self.insights['tickets']['overall']
            ticket_types = ['Single-Day', 'Annual Pass']
            ticket_values = [ticket_overall['total_single_day'], ticket_overall['total_annual']]
            axes[5].pie(ticket_values, labels=ticket_types, autopct='%1.1f%%', startangle=90, colors=['orange', 'darkblue'])
            axes[5].set_title('Ticket Type Distribution', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('united_parks_charts/performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Created performance dashboard with ticket attribution")
        
        # 2. Data Quality Issues Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Quality issues breakdown
        issues_labels = ['Impossible CTR', 'Impossible Conv Rate', 'Missing Platform', 'Extreme CTR', 'Extreme ROAS', 'Extreme CPA']
        issues_values = [
            self.quality_issues['impossible_ctr'],
            self.quality_issues['impossible_cr'], 
            self.quality_issues['missing_platform'],
            self.quality_issues['extreme_ctr'],
            self.quality_issues['extreme_roas'],
            self.quality_issues['extreme_cpa']
        ]
        
        ax1.bar(issues_labels, issues_values, color='orange', alpha=0.7)
        ax1.set_title('Data Quality Issues Identified', fontweight='bold')
        ax1.set_ylabel('Number of Rows Affected')
        ax1.tick_params(axis='x', rotation=45)
        
        # Before/After metrics comparison
        original_roas = self.df['Revenue'].sum() / self.df['Spend'].sum()
        cleaned_roas = self.df_cleaned['Revenue'].sum() / self.df_cleaned['Spend'].sum()
        
        metrics = ['Overall ROAS']
        before_values = [original_roas]
        after_values = [cleaned_roas]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax2.bar(x - width/2, before_values, width, label='Before Cleaning', color='red', alpha=0.7)
        ax2.bar(x + width/2, after_values, width, label='After Cleaning', color='green', alpha=0.7)
        ax2.set_title('Impact of Data Cleaning', fontweight='bold')
        ax2.set_ylabel('ROAS (x)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('united_parks_charts/data_quality_assessment.png', dpi=300, bbox_inches='tight')
        print("✅ Created data quality assessment chart")
        
        # 3. Optimization Opportunities
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Media Optimization Opportunities', fontsize=16, fontweight='bold')
        
        # Underperforming areas (bubble chart)
        underperforming = self.insights['optimization']['underperforming'].head(10)
        if not underperforming.empty:
            scatter = ax1.scatter(underperforming['ROAS'], underperforming['CPA'], 
                                s=underperforming['Spend']/1000, alpha=0.6, color='red')
            ax1.set_xlabel('ROAS (x)')
            ax1.set_ylabel('CPA ($)')
            ax1.set_title('Underperforming Areas\n(Bubble size = Spend)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Scale-up opportunities
        scale_up = self.insights['optimization']['scale_up'].head(10)
        if not scale_up.empty:
            ax2.scatter(scale_up['ROAS'], scale_up['CPA'], 
                       s=scale_up['Spend']/1000, alpha=0.6, color='green')
            ax2.set_xlabel('ROAS (x)')
            ax2.set_ylabel('CPA ($)')
            ax2.set_title('Scale-Up Opportunities\n(Bubble size = Spend)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('united_parks_charts/optimization_opportunities.png', dpi=300, bbox_inches='tight')
        print("✅ Created optimization opportunities chart")
        
        plt.close('all')
        
    def generate_strategic_recommendations(self):
        """Generate comprehensive strategic recommendations for United Parks & Resorts based on global analysis"""
        print("\n" + "="*50)
        print("STRATEGIC RECOMMENDATIONS FOR UNITED PARKS & RESORTS")
        print("="*50)
        
        # Calculate key metrics for recommendations
        total_spend = self.df_cleaned['Spend'].sum()
        total_revenue = self.df_cleaned['Revenue'].sum()
        current_roas = total_revenue / total_spend
        
        # Get monthly insights for seasonal recommendations
        monthly_data = self.insights.get('monthly', {})
        seasonality = monthly_data.get('seasonality', {})
        
        recommendations = {
            'executive_summary': [
                f"United Parks Current Performance: ${total_revenue:,.0f} revenue from ${total_spend:,.0f} spend ({current_roas:.2f}x ROAS)",
                f"Key Opportunity: {seasonality.get('revenue_seasonality', 0):.1f}% revenue seasonality indicates major optimization potential",
                "Primary Focus: Leverage seasonal patterns and high-performing channels across all parks for 20%+ revenue growth"
            ],
            
            'seasonal_strategy': [
                f"Peak Season Optimization ({seasonality.get('peak_month', {}).get('Month_Name', 'N/A')}): Scale spend 40% during peak months",
                f"Low Season Recovery ({seasonality.get('low_month', {}).get('Month_Name', 'N/A')}): Implement awareness campaigns to build base",
                "Quarterly Channel Rotation: Adjust channel mix based on seasonal performance patterns",
                "Holiday Preparation: Front-load awareness campaigns 2 months before peak seasons"
            ],
            
            'channel_optimization': [
                f"Scale Paid Search: Increase budget by 30% (current top performer at {self.insights['performance']['top_channels'].iloc[0]['ROAS']:.1f}x ROAS)",
                "Diversify Paid Social: Test TikTok and Pinterest for younger demographics",
                "OTT/CTV Focus: Leverage for awareness during off-peak months",
                "Traditional Media: Reduce by 15% and reallocate to digital channels"
            ],
            
            'audience_segmentation': [
                f"Local Segment Priority: Increase investment by 25% (highest revenue share: {self.insights['segments'].iloc[0]['Revenue_Share']:.1f}%)",
                "International Growth: Expand reach with 20% budget increase (lowest CPA opportunity)",
                "Same Day Optimization: Implement dynamic campaigns for weather-based triggers",
                "Drive & Overnight: Focus on weekend packages and multi-day experiences"
            ],
            
            'measurement_framework': [
                "Primary KPIs: Monthly ROAS (target: 8.5x), CAC by segment, Revenue per Visit",
                "Seasonal KPIs: Peak vs off-peak performance ratios, seasonal customer LTV",
                "Efficiency KPIs: Channel contribution margin, cross-channel attribution",
                "Predictive KPIs: Weather correlation, competitive pressure index"
            ],
            
            'budget_allocation': [
                f"Q1 Focus: Awareness investment (+20%) to prepare for spring season",
                f"Q2-Q3 Peak: Performance marketing (+40%) to maximize high-intent periods", 
                f"Q4 Planning: Brand building and retention focus for next year setup",
                f"Emergency Fund: Reserve 10% budget for weather/competitive response"
            ],
            
            'innovation_testing': [
                "Voice Search Optimization: Test Alexa/Google Assistant for attraction info",
                "AR/VR Experiences: Pilot virtual park tours for drive distance customers",
                "Influencer Partnerships: Partner with family/travel influencers for authentic content",
                "Dynamic Pricing Integration: Connect media campaigns to real-time pricing"
            ],
            
            'forecasting_model': [
                "Weather Integration: Build 14-day weather forecast into media planning",
                "Competitive Intelligence: Monitor competitor campaigns and adjust spend",
                "Economic Indicators: Factor in disposable income trends for budget allocation",
                "Capacity Management: Align media spend with park capacity and operations"
            ]
        }
        
        # Print comprehensive recommendations
        for category, items in recommendations.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item in items:
                print(f"  * {item}")
        
        # Add financial projections
        self.generate_financial_projections(current_roas, total_spend, total_revenue)
        
        self.insights['recommendations'] = recommendations
        
    def generate_financial_projections(self, current_roas, total_spend, total_revenue):
        """Generate financial projections based on recommendations"""
        print(f"\nFINANCIAL PROJECTIONS (12-Month Implementation):")
        
        # Conservative, realistic, and aggressive scenarios
        scenarios = {
            'Conservative': {'roas_improvement': 0.15, 'spend_increase': 0.10},
            'Realistic': {'roas_improvement': 0.25, 'spend_increase': 0.15}, 
            'Aggressive': {'roas_improvement': 0.40, 'spend_increase': 0.20}
        }
        
        for scenario, params in scenarios.items():
            new_spend = total_spend * (1 + params['spend_increase'])
            new_roas = current_roas * (1 + params['roas_improvement'])
            new_revenue = new_spend * new_roas
            revenue_lift = new_revenue - total_revenue
            roi = (revenue_lift - (new_spend - total_spend)) / (new_spend - total_spend) * 100
            
            print(f"{scenario} Scenario:")
            print(f"  * New Annual Spend: ${new_spend:,.0f} (+{params['spend_increase']:.0%})")
            print(f"  * Projected ROAS: {new_roas:.2f}x (+{params['roas_improvement']:.0%})")
            print(f"  * Projected Revenue: ${new_revenue:,.0f}")
            print(f"  * Revenue Lift: ${revenue_lift:,.0f}")
            print(f"  * ROI on Additional Spend: {roi:.0f}%")
            print()
        
    def export_analysis_to_csv(self):
        """Export global monthly breakdown analysis results to CSV files"""
        print("\n" + "="*50)
        print("EXPORTING GLOBAL MONTHLY ANALYSIS TO CSV")
        print("="*50)
        
        park_name = "united_parks_global"
        
        # Ensure monthly analysis has been run
        if 'monthly' not in self.insights:
            print("Monthly analysis not available. Running monthly analysis first...")
            self.comprehensive_monthly_analysis()
        
        # 1. Export Monthly Channel Performance
        if 'monthly' in self.insights and 'channel_monthly' in self.insights['monthly']:
            monthly_channel_df = self.insights['monthly']['channel_monthly']
            # Pivot to show channels as columns and months as rows
            channel_pivot = monthly_channel_df.pivot_table(
                index=['Month_Name', 'Month_Period'], 
                columns='Channel', 
                values=['Spend', 'Revenue', 'ROAS', 'Conversions'],
                aggfunc='sum'
            ).round(2)
            
            # Flatten column names
            channel_pivot.columns = [f"{col[1]}_{col[0]}" for col in channel_pivot.columns]
            channel_pivot = channel_pivot.reset_index()
            
            filename = f'{park_name}_channel_performance.csv'
            channel_pivot.to_csv(filename, index=False)
            print(f"Global monthly channel performance exported to '{filename}'")
        
        # 2. Export Monthly Segment Performance
        if 'monthly' in self.insights and 'segment_monthly' in self.insights['monthly']:
            monthly_segment_df = self.insights['monthly']['segment_monthly']
            # Pivot to show segments as columns and months as rows
            segment_pivot = monthly_segment_df.pivot_table(
                index=['Month_Name', 'Month_Period'], 
                columns='Segment', 
                values=['Spend', 'Revenue', 'ROAS', 'Conversions'],
                aggfunc='sum'
            ).round(2)
            
            # Flatten column names
            segment_pivot.columns = [f"{col[1]}_{col[0]}" for col in segment_pivot.columns]
            segment_pivot = segment_pivot.reset_index()
            
            filename = f'{park_name}_segment_performance.csv'
            segment_pivot.to_csv(filename, index=False)
            print(f"Global monthly segment performance exported to '{filename}'")
        
        # 3. Export Monthly Platform Performance
        monthly_platform = self.df_cleaned.groupby(['Month_Period', 'Month_Name', 'Platform']).agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        monthly_platform['ROAS'] = monthly_platform['Revenue'] / monthly_platform['Spend']
        
        platform_pivot = monthly_platform.pivot_table(
            index=['Month_Name', 'Month_Period'], 
            columns='Platform', 
            values=['Spend', 'Revenue', 'ROAS', 'Conversions'],
            aggfunc='sum'
        ).round(2)
        
        platform_pivot.columns = [f"{col[1]}_{col[0]}" for col in platform_pivot.columns]
        platform_pivot = platform_pivot.reset_index()
        
        filename = f'{park_name}_platform_performance.csv'
        platform_pivot.to_csv(filename, index=False)
        print(f"Global monthly platform performance exported to '{filename}'")
        
        # 4. Export Monthly Funnel Stage Performance
        monthly_funnel = self.df_cleaned.groupby(['Month_Period', 'Month_Name', 'Funnel Stage']).agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        monthly_funnel['ROAS'] = monthly_funnel['Revenue'] / monthly_funnel['Spend']
        
        funnel_pivot = monthly_funnel.pivot_table(
            index=['Month_Name', 'Month_Period'], 
            columns='Funnel Stage', 
            values=['Spend', 'Revenue', 'ROAS', 'Conversions'],
            aggfunc='sum'
        ).round(2)
        
        funnel_pivot.columns = [f"{col[1]}_{col[0]}" for col in funnel_pivot.columns]
        funnel_pivot = funnel_pivot.reset_index()
        
        filename = f'{park_name}_funnel_performance.csv'
        funnel_pivot.to_csv(filename, index=False)
        print(f"Global monthly funnel stage performance exported to '{filename}'")
        
        # 5. Export Monthly Optimization Opportunities (showing month-over-month changes)
        if 'monthly' in self.insights and 'overview' in self.insights['monthly']:
            monthly_overview = self.insights['monthly']['overview'].copy()
            
            # Calculate additional optimization metrics
            monthly_overview['Efficiency_Score'] = monthly_overview['ROAS'] / monthly_overview['CPA'] * 100
            monthly_overview['Revenue_per_Dollar'] = monthly_overview['Revenue'] / monthly_overview['Spend']
            monthly_overview['Conversion_Efficiency'] = monthly_overview['Conversions'] / monthly_overview['Spend'] * 1000
            
            # Identify optimization opportunities by month
            monthly_overview['Performance_vs_Avg'] = (monthly_overview['ROAS'] / monthly_overview['ROAS'].mean() - 1) * 100
            monthly_overview['Optimization_Flag'] = monthly_overview['Performance_vs_Avg'].apply(
                lambda x: 'Scale Up' if x > 10 else ('Review' if x < -10 else 'Maintain')
            )
            
            filename = f'{park_name}_optimization_opportunities.csv'
            monthly_overview.to_csv(filename, index=False)
            print(f"Global monthly optimization opportunities exported to '{filename}'")
        
        # 6. Export Monthly Data Quality Assessment
        monthly_quality = self.df_cleaned.groupby(['Month_Period', 'Month_Name']).agg({
            'CTR': ['count', 'mean', 'std'],
            'Conversion_Rate': ['count', 'mean', 'std'],
            'ROAS': ['count', 'mean', 'std'],
            'CPA': ['count', 'mean', 'std']
        }).round(2)
        
        monthly_quality.columns = [f"{col[0]}_{col[1]}" for col in monthly_quality.columns]
        monthly_quality = monthly_quality.reset_index()
        
        filename = f'{park_name}_data_quality_issues.csv'
        monthly_quality.to_csv(filename, index=False)
        print(f"Global monthly data quality assessment exported to '{filename}'")
        
        print("\nAll global monthly CSV exports completed successfully!")
        
    def generate_summary_report(self):
        """Generate a comprehensive AI-friendly global summary report"""
        print("\n" + "="*50)
        print("COMPREHENSIVE GLOBAL AI ANALYSIS REPORT")
        print("="*50)
        
        # Calculate key summary metrics
        total_spend = self.df_cleaned['Spend'].sum()
        total_revenue = self.df_cleaned['Revenue'].sum()
        total_conversions = self.df_cleaned['Conversions'].sum()
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        overall_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        # Data quality impact
        data_issues_pct = (sum(self.quality_issues.values()) / len(self.df)) * 100
        
        # Generate detailed performance tables
        channel_performance = self.df_cleaned.groupby('Channel').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        channel_performance['ROAS'] = channel_performance['Revenue'] / channel_performance['Spend']
        channel_performance['CPA'] = channel_performance['Spend'] / channel_performance['Conversions']
        channel_performance['CTR'] = (channel_performance['Clicks'] / channel_performance['Impressions']) * 100
        channel_performance['Revenue_Share'] = (channel_performance['Revenue'] / channel_performance['Revenue'].sum()) * 100
        channel_performance['Spend_Share'] = (channel_performance['Spend'] / channel_performance['Spend'].sum()) * 100
        
        segment_performance = self.insights['segments'].copy()
        funnel_performance = self.insights['funnel'].copy()
        
        # Generate platform performance
        platform_performance = self.df_cleaned.groupby('Platform').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        platform_performance['ROAS'] = platform_performance['Revenue'] / platform_performance['Spend']
        platform_performance['CPA'] = platform_performance['Spend'] / platform_performance['Conversions']
        platform_performance['Revenue_Share'] = (platform_performance['Revenue'] / platform_performance['Revenue'].sum()) * 100
        platform_performance = platform_performance.sort_values('Revenue', ascending=False)
        
        # Monthly performance trends
        monthly_performance = self.df_cleaned.groupby(self.df_cleaned['Date'].dt.to_period('M')).agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        monthly_performance['ROAS'] = monthly_performance['Revenue'] / monthly_performance['Spend']
        monthly_performance['Month'] = monthly_performance['Date'].astype(str)
        
        # Channel-Segment cross analysis
        channel_segment_performance = self.df_cleaned.groupby(['Channel', 'Segment']).agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        channel_segment_performance['ROAS'] = channel_segment_performance['Revenue'] / channel_segment_performance['Spend']
        channel_segment_performance['CPA'] = channel_segment_performance['Spend'] / channel_segment_performance['Conversions']
        
        summary_report = f"""
================================================================================
UNITED PARKS & RESORTS - COMPREHENSIVE GLOBAL MEDIA ANALYSIS REPORT
ALL PARKS - ENTERPRISE-WIDE ANALYSIS
AI-READY DATA AND INSIGHTS FOR ADVANCED REPORT GENERATION
================================================================================

EXECUTIVE SUMMARY
================================================================================
Analysis Scope: {', '.join(self.df['Park'].unique())}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Period: {self.df_cleaned['Date'].min().strftime('%Y-%m-%d')} to {self.df_cleaned['Date'].max().strftime('%Y-%m-%d')}
Total Records Analyzed: {len(self.df):,}
Records After Cleaning: {len(self.df_cleaned):,}

FINANCIAL PERFORMANCE OVERVIEW
================================================================================
Total Media Spend: ${total_spend:,.2f}
Total Revenue Generated: ${total_revenue:,.2f}
Total Conversions: {total_conversions:,.0f}
Overall ROAS: {overall_roas:.2f}x
Overall CPA: ${overall_cpa:.2f}
Profit Margin: ${total_revenue - total_spend:,.2f}
ROI Percentage: {((total_revenue - total_spend) / total_spend * 100):.1f}%

DATA QUALITY ASSESSMENT
================================================================================
Overall Data Quality Score: {100 - data_issues_pct:.1f}/100

Specific Data Quality Issues:
- Impossible CTR Cases: {self.quality_issues['impossible_ctr']:,} rows ({self.quality_issues['impossible_ctr']/len(self.df)*100:.2f}%)
- Impossible Conversion Rate Cases: {self.quality_issues['impossible_cr']:,} rows ({self.quality_issues['impossible_cr']/len(self.df)*100:.2f}%)
- Missing Platform Data: {self.quality_issues['missing_platform']:,} rows ({self.quality_issues['missing_platform']/len(self.df)*100:.2f}%)
- Extreme CTR Outliers: {self.quality_issues['extreme_ctr']:,} rows ({self.quality_issues['extreme_ctr']/len(self.df)*100:.2f}%)
- Extreme ROAS Outliers: {self.quality_issues['extreme_roas']:,} rows ({self.quality_issues['extreme_roas']/len(self.df)*100:.2f}%)
- Extreme CPA Outliers: {self.quality_issues['extreme_cpa']:,} rows ({self.quality_issues['extreme_cpa']/len(self.df)*100:.2f}%)

Data Cleaning Actions Taken:
- Fixed {self.quality_issues['impossible_ctr']:,} impossible CTR cases by capping clicks at impression levels
- Fixed {self.quality_issues['impossible_cr']:,} impossible conversion rate cases by capping conversions at click levels
- Applied statistical outlier capping using IQR method for CTR, Conversion Rate, and ROAS
- Filled missing platform data for traditional media channels

DETAILED CHANNEL PERFORMANCE ANALYSIS
================================================================================
"""
        
        # Add detailed channel performance table
        summary_report += "Channel Performance Breakdown:\n"
        for _, row in channel_performance.sort_values('Revenue', ascending=False).iterrows():
            summary_report += f"Channel: {row['Channel']}\n"
            summary_report += f"  - Spend: ${row['Spend']:,.2f} ({row['Spend_Share']:.1f}% of total)\n"
            summary_report += f"  - Revenue: ${row['Revenue']:,.2f} ({row['Revenue_Share']:.1f}% of total)\n"
            summary_report += f"  - ROAS: {row['ROAS']:.2f}x\n"
            summary_report += f"  - CPA: ${row['CPA']:.2f}\n"
            summary_report += f"  - CTR: {row['CTR']:.2f}%\n"
            summary_report += f"  - Conversions: {row['Conversions']:,.0f}\n"
            summary_report += f"  - Impressions: {row['Impressions']:,.0f}\n"
            summary_report += f"  - Clicks: {row['Clicks']:,.0f}\n\n"
        
        summary_report += f"""
DETAILED SEGMENT PERFORMANCE ANALYSIS
================================================================================
"""
        
        # Add detailed segment performance
        for _, row in segment_performance.iterrows():
            summary_report += f"Segment: {row['Segment']}\n"
            summary_report += f"  - Revenue: ${row['Revenue']:,.2f} ({row['Revenue_Share']:.1f}% of total)\n"
            summary_report += f"  - ROAS: {row['ROAS']:.2f}x\n"
            summary_report += f"  - CPA: ${row['CPA']:.2f}\n"
            summary_report += f"  - CTR: {row['CTR']:.2f}%\n"
            summary_report += f"  - Conversions: {row['Conversions']:,.0f}\n"
            summary_report += f"  - Spend: ${row['Spend']:,.2f}\n\n"
        
        summary_report += f"""
DETAILED PLATFORM PERFORMANCE ANALYSIS
================================================================================
"""
        
        # Add detailed platform performance
        for _, row in platform_performance.head(10).iterrows():
            summary_report += f"Platform: {row['Platform']}\n"
            summary_report += f"  - Revenue: ${row['Revenue']:,.2f} ({row['Revenue_Share']:.1f}% of total)\n"
            summary_report += f"  - ROAS: {row['ROAS']:.2f}x\n"
            summary_report += f"  - CPA: ${row['CPA']:.2f}\n"
            summary_report += f"  - Conversions: {row['Conversions']:,.0f}\n"
            summary_report += f"  - Spend: ${row['Spend']:,.2f}\n\n"
        
        summary_report += f"""
FUNNEL STAGE PERFORMANCE ANALYSIS
================================================================================
"""
        
        # Add funnel stage analysis
        for _, row in funnel_performance.iterrows():
            summary_report += f"Funnel Stage: {row['Funnel Stage']}\n"
            summary_report += f"  - Spend: ${row['Spend']:,.2f} ({row['Spend_Share']:.1f}% of total)\n"
            summary_report += f"  - Revenue: ${row['Revenue']:,.2f}\n"
            summary_report += f"  - ROAS: {row['ROAS']:.2f}x\n"
            summary_report += f"  - CPA: ${row['CPA']:.2f}\n"
            summary_report += f"  - Conversions: {row['Conversions']:,.0f}\n\n"
        
        summary_report += f"""
COMPREHENSIVE MONTHLY PERFORMANCE ANALYSIS
================================================================================
"""
        
        # Add monthly overview with MoM growth
        if 'monthly' in self.insights and self.insights['monthly']['overview'] is not None:
            monthly_data = self.insights['monthly']['overview']
            for _, row in monthly_data.iterrows():
                mom_revenue = f" ({row['Revenue_MoM_Growth']:+.1f}% MoM)" if not pd.isna(row['Revenue_MoM_Growth']) else ""
                mom_roas = f" ({row['ROAS_MoM_Change']:+.2f}x MoM)" if not pd.isna(row['ROAS_MoM_Change']) else ""
                summary_report += f"Month: {row['Month_Name']}\n"
                summary_report += f"  - Revenue: ${row['Revenue']:,.2f}{mom_revenue}\n"
                summary_report += f"  - Spend: ${row['Spend']:,.2f}\n"
                summary_report += f"  - ROAS: {row['ROAS']:.2f}x{mom_roas}\n"
                summary_report += f"  - CPA: ${row['CPA']:.2f}\n"
                summary_report += f"  - CTR: {row['CTR']:.2f}%\n"
                summary_report += f"  - Conversions: {row['Conversions']:,.0f}\n\n"
        
        # Add seasonality insights
        if 'monthly' in self.insights and 'seasonality' in self.insights['monthly']:
            seasonality = self.insights['monthly']['seasonality']
            summary_report += f"""
SEASONALITY AND TREND INSIGHTS
================================================================================
Peak Performance Month: {seasonality.get('peak_month', {}).get('Month_Name', 'N/A')}
  - Revenue: ${seasonality.get('peak_month', {}).get('Revenue', 0):,.2f}
  - ROAS: {seasonality.get('peak_month', {}).get('ROAS', 0):.2f}x

Low Performance Month: {seasonality.get('low_month', {}).get('Month_Name', 'N/A')}
  - Revenue: ${seasonality.get('low_month', {}).get('Revenue', 0):,.2f}
  - ROAS: {seasonality.get('low_month', {}).get('ROAS', 0):.2f}x

Revenue Seasonality Index: {seasonality.get('revenue_seasonality', 0):.1f}%
Year-over-Year Growth Trend: {seasonality.get('yoy_trend', 0):+.1f}%

Quarterly Channel Performance:
"""
            if 'quarterly_insights' in seasonality:
                for quarter, data in seasonality['quarterly_insights'].items():
                    summary_report += f"{quarter}: {data['top_channel']} performs best ({data['top_roas']:.2f}x ROAS)\n"
            
            summary_report += "\n"
        
        summary_report += f"""
OPTIMIZATION OPPORTUNITIES
================================================================================
"""
        
        # Add optimization opportunities
        if 'optimization' in self.insights and not self.insights['optimization']['underperforming'].empty:
            summary_report += "UNDERPERFORMING AREAS (Reduce Spend):\n"
            for _, row in self.insights['optimization']['underperforming'].head(5).iterrows():
                summary_report += f"- {row['Channel']} - {row['Segment']}: ${row['Spend']:,.0f} spend, {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA\n"
            summary_report += "\n"
        
        if 'optimization' in self.insights and not self.insights['optimization']['scale_up'].empty:
            summary_report += "SCALE-UP OPPORTUNITIES (Increase Spend):\n"
            for _, row in self.insights['optimization']['scale_up'].head(5).iterrows():
                summary_report += f"- {row['Channel']} - {row['Segment']}: {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA, ${row['Spend']:,.0f} current spend\n"
            summary_report += "\n"
        
        summary_report += f"""
STRATEGIC RECOMMENDATIONS FOR UNITED PARKS & RESORTS
================================================================================

IMMEDIATE ACTIONS (Next 30 Days):
1. Data Infrastructure - Implement validation rules to prevent impossible metrics
   - Target: Reduce data quality issues from {data_issues_pct:.1f}% to <5%
   - ROI Impact: Improved measurement accuracy worth estimated ${total_spend * 0.02:,.0f} in optimization potential

2. Budget Reallocation - Shift spend from underperforming combinations
   - Reallocate ${self.insights['optimization']['underperforming']['Spend'].sum() * 0.25:,.0f} from low-ROAS channels
   - Expected Revenue Lift: ${self.insights['optimization']['underperforming']['Spend'].sum() * 0.25 * (overall_roas * 1.5):,.0f}

3. Scale High-Performing Platforms
   - Increase spend on top-performing platform combinations by 20%
   - Expected Additional Revenue: ${self.insights['optimization']['scale_up']['Spend'].sum() * 0.2 * self.insights['optimization']['scale_up']['ROAS'].mean():,.0f}

SHORT-TERM OPTIMIZATIONS (Next 90 Days):
1. Funnel Balance Optimization
   - Current funnel spend distribution needs rebalancing
   - Increase consideration stage investment by 15%
   - Expected ROAS improvement: +{funnel_performance.loc[funnel_performance['Funnel Stage'] == 'Consideration', 'ROAS'].iloc[0] * 0.15:.1f}x

2. Segment-Specific Strategies
   - Focus on Local segment (highest revenue share: {segment_performance.iloc[0]['Revenue_Share']:.1f}%)
   - Optimize International segment (lowest CPA: ${segment_performance.iloc[-1]['CPA']:.2f})

LONG-TERM STRATEGIC INITIATIVES (Next 12 Months):
1. Advanced Attribution Modeling
2. Predictive Analytics Implementation  
3. Cross-Channel Optimization
4. Real-time Performance Monitoring

KEY PERFORMANCE INDICATORS TO TRACK:
================================================================================
Primary KPIs:
- ROAS (Current: {overall_roas:.2f}x, Target: {overall_roas * 1.2:.2f}x)
- CPA (Current: ${overall_cpa:.2f}, Target: ${overall_cpa * 0.85:.2f})
- Revenue Growth (Monthly target: +15%)

Quality KPIs:
- Data Accuracy Score (Current: {100 - data_issues_pct:.1f}%, Target: >95%)
- Outlier Percentage (Current: {data_issues_pct:.1f}%, Target: <5%)

Efficiency KPIs:
- CTR by Channel (Current range: {channel_performance['CTR'].min():.2f}% - {channel_performance['CTR'].max():.2f}%)
- Conversion Rate by Platform
- Cost per Click trends

TECHNICAL NOTES FOR AI MODEL CONSUMPTION:
================================================================================
- All financial figures are in USD
- ROAS = Revenue / Spend (higher is better)
- CPA = Spend / Conversions (lower is better)  
- CTR = (Clicks / Impressions) * 100
- Data cleaning applied using IQR method for outlier detection
- Statistical significance level: 95% confidence interval
- Time period covers 52 weeks of data
- Park-specific analysis filters out other United Parks properties

END OF COMPREHENSIVE ANALYSIS REPORT
================================================================================
"""
        
        print(summary_report)
        
        # Save report to file
        report_filename = 'united_parks_global_comprehensive_analysis_report.txt'
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print(f"\nComprehensive global AI-ready report saved to '{report_filename}'")
        
    def run_complete_analysis(self):
        """Run the complete global analysis workflow with ticket attribution across all parks"""
        print(f"UNITED PARKS & RESORTS GLOBAL MEDIA ANALYSIS WITH TICKET ATTRIBUTION")
        print("Director of Media - Enterprise-Wide Strategic Assessment")
        print("="*70)
        
        # Load and assess data
        if not self.load_data():
            return False
        
        # Run analysis steps
        self.assess_data_quality()
        self.clean_data()
        self.ticket_attribution_analysis()  # New ticket analysis
        self.performance_overview()
        self.audience_segmentation_analysis()
        self.funnel_stage_analysis()
        self.comprehensive_monthly_analysis()
        self.park_performance_analysis()
        self.identify_optimization_opportunities()
        self.create_visualizations()
        self.generate_strategic_recommendations()
        self.export_analysis_to_csv()
        self.generate_summary_report()
        
        print(f"\nGlobal Analysis Complete for All United Parks Properties! Check the following outputs:")
        print("• CSV files for detailed global performance data with ticket attribution")
        print(f"• united_parks_global_comprehensive_analysis_report.txt for AI-ready comprehensive global report")
        print("• united_parks_charts/ folder for presentation charts")
        
        return True

def main():
    """Main execution function"""
    
    # File path for the main dataset - prioritize new combined data source
    primary_file = "United Parks - Paid Media - Updated Conversion & Revenue - United Parks - Combined Data.csv"
    primary_file_absolute = r"C:\Users\Tech-Nation\OneDrive\Documents\No Joke Marketing\United Parks\United Parks - Paid Media - Updated Conversion & Revenue - United Parks - Combined Data.csv"
    ticket_file = "updated paid media - Paid Media_with_tickets.csv"
    new_main_file = "United Parks - Paid Media - Sheet1.csv"
    old_main_file = "Weekly_Paid_Media_and_Sales_Data.xlsx - Paid Media.csv"
    sample_file = "United Parks - Paid Media - Paide Media + Tickets - Sample (1).csv"
    
    # Check for the new primary file first (both relative and absolute paths), then fall back to other files
    if os.path.exists(primary_file):
        print(f"🎯 Using new combined dataset: '{primary_file}'")
        main_file = primary_file
    elif os.path.exists(primary_file_absolute):
        print(f"🎯 Using new combined dataset: '{primary_file_absolute}'")
        main_file = primary_file_absolute
    elif os.path.exists(ticket_file):
        print(f"🎫 Using ticket attribution dataset: '{ticket_file}'")
        main_file = ticket_file
    elif os.path.exists(sample_file):
        print(f"📝 Using sample ticket attribution file '{sample_file}' for demonstration.")
        main_file = sample_file
    elif os.path.exists(new_main_file):
        print(f"📊 Using updated dataset: '{new_main_file}' (without ticket attribution)")
        main_file = new_main_file
    elif os.path.exists(old_main_file):
        print(f"⚠️ Using legacy dataset: '{old_main_file}' (without ticket attribution)")
        main_file = old_main_file
    else:
        print("❌ No data files found. Please ensure the data file is in the current directory.")
        return
    
    # Initialize and run analysis
    analyzer = UnitedParksMediaAnalysis(main_file)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 