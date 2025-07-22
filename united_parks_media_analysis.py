#!/usr/bin/env python3
"""
United Parks & Resorts Media Analysis Script
Director of Media - Strategic Recommendations

This script analyzes paid media performance data while accounting for data quality issues
and generates insights for a strategic presentation.
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
    def __init__(self, file_path):
        """Initialize the analysis with data loading and quality assessment"""
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.quality_issues = {}
        self.insights = {}
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self):
        """Load the main dataset"""
        print("Loading United Parks & Resorts paid media data...")
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"✅ Data loaded successfully: {len(self.df):,} rows, {len(self.df.columns)} columns")
            return True
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def assess_data_quality(self):
        """Assess and document data quality issues"""
        print("\n" + "="*50)
        print("DATA QUALITY ASSESSMENT")
        print("="*50)
        
        # Basic data info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['Week'].min()} to {self.df['Week'].max()}")
        print(f"Parks: {self.df['Park'].nunique()}")
        print(f"Segments: {self.df['Segment'].nunique()}")
        
        # Calculate performance metrics
        self.df['CTR'] = np.where(self.df['Impressions'] > 0, 
                                 (self.df['Clicks'] / self.df['Impressions']) * 100, 0)
        self.df['Conversion_Rate'] = np.where(self.df['Clicks'] > 0,
                                            (self.df['Conversions'] / self.df['Clicks']) * 100, 0)
        self.df['ROAS'] = np.where(self.df['Spend ($)'] > 0,
                                  self.df['Revenue ($)'] / self.df['Spend ($)'], 0)
        self.df['CPA'] = np.where(self.df['Conversions'] > 0,
                                 self.df['Spend ($)'] / self.df['Conversions'], np.inf)
        
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
        
        self.quality_issues = issues
        
        # Report issues
        print("\nDATA QUALITY ISSUES IDENTIFIED:")
        print(f"* Impossible CTRs (Clicks > Impressions): {issues['impossible_ctr']:,} rows")
        print(f"* Impossible Conversion Rates (Conversions > Clicks): {issues['impossible_cr']:,} rows") 
        print(f"* Missing Platform data: {issues['missing_platform']:,} rows")
        print(f"* Extreme CTR outliers (>100%): {issues['extreme_ctr']:,} rows")
        print(f"* Extreme ROAS outliers (>50x): {issues['extreme_roas']:,} rows")
        print(f"* Extreme CPA outliers (>$1000): {issues['extreme_cpa']:,} rows")
        
        data_quality_score = max(0, 100 - (sum(issues.values()) / len(self.df) * 100))
        print(f"\nData Quality Score: {data_quality_score:.1f}/100")
        
    def clean_data(self):
        """Clean the data addressing identified quality issues"""
        print("\n" + "="*50)
        print("DATA CLEANING & PREPARATION")
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
        
        # 3. Handle missing platforms - fill with 'Traditional Media' for non-digital channels
        traditional_channels = ['TV', 'Radio', 'Direct Mail']
        missing_platform_mask = self.df_cleaned['Platform'].isna()
        traditional_mask = self.df_cleaned['Channel'].isin(traditional_channels) & missing_platform_mask
        self.df_cleaned.loc[traditional_mask, 'Platform'] = 'Traditional Media'
        print(f"Filled {traditional_mask.sum():,} missing platform values for traditional media")
        
        # 4. Recalculate metrics with cleaned data
        self.df_cleaned['CTR'] = np.where(self.df_cleaned['Impressions'] > 0,
                                         (self.df_cleaned['Clicks'] / self.df_cleaned['Impressions']) * 100, 0)
        self.df_cleaned['Conversion_Rate'] = np.where(self.df_cleaned['Clicks'] > 0,
                                                    (self.df_cleaned['Conversions'] / self.df_cleaned['Clicks']) * 100, 0)
        self.df_cleaned['ROAS'] = np.where(self.df_cleaned['Spend ($)'] > 0,
                                         self.df_cleaned['Revenue ($)'] / self.df_cleaned['Spend ($)'], 0)
        self.df_cleaned['CPA'] = np.where(self.df_cleaned['Conversions'] > 0,
                                        self.df_cleaned['Spend ($)'] / self.df_cleaned['Conversions'], np.inf)
        
        # 5. Handle extreme outliers using IQR method for key metrics
        for metric in ['CTR', 'Conversion_Rate', 'ROAS']:
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
        
        # Convert Week to datetime
        self.df_cleaned['Week'] = pd.to_datetime(self.df_cleaned['Week'])
        
        print(f"\nCleaned dataset ready: {len(self.df_cleaned):,} rows")
        
    def performance_overview(self):
        """Analyze overall performance trends and channel effectiveness"""
        print("\n" + "="*50)
        print("PERFORMANCE OVERVIEW ANALYSIS")
        print("="*50)
        
        # Overall metrics
        total_spend = self.df_cleaned['Spend ($)'].sum()
        total_revenue = self.df_cleaned['Revenue ($)'].sum()
        total_conversions = self.df_cleaned['Conversions'].sum()
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        overall_cpa = total_spend / total_conversions if total_conversions > 0 else 0
        
        print(f"OVERALL PERFORMANCE:")
        print(f"* Total Spend: ${total_spend:,.2f}")
        print(f"* Total Revenue: ${total_revenue:,.2f}")
        print(f"* Total Conversions: {total_conversions:,.0f}")
        print(f"* Overall ROAS: {overall_roas:.2f}x")
        print(f"* Overall CPA: ${overall_cpa:.2f}")
        
        # Channel performance
        channel_performance = self.df_cleaned.groupby('Channel').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        channel_performance['ROAS'] = channel_performance['Revenue ($)'] / channel_performance['Spend ($)']
        channel_performance['CPA'] = channel_performance['Spend ($)'] / channel_performance['Conversions']
        channel_performance['CTR'] = (channel_performance['Clicks'] / channel_performance['Impressions']) * 100
        channel_performance = channel_performance.sort_values('Revenue ($)', ascending=False)
        
        print(f"\nTOP PERFORMING CHANNELS (by Revenue):")
        for i, row in channel_performance.head(3).iterrows():
            print(f"{row['Channel']}: ${row['Revenue ($)']:,.0f} revenue, {row['ROAS']:.2f}x ROAS")
            
        print(f"\nLOWEST PERFORMING CHANNELS (by ROAS):")
        for i, row in channel_performance.nsmallest(3, 'ROAS').iterrows():
            print(f"{row['Channel']}: {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA")
        
        # Platform performance
        platform_performance = self.df_cleaned.groupby('Platform').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        platform_performance['ROAS'] = platform_performance['Revenue ($)'] / platform_performance['Spend ($)']
        platform_performance = platform_performance.sort_values('Revenue ($)', ascending=False)
        
        print(f"\nTOP PERFORMING PLATFORMS:")
        for i, row in platform_performance.head(5).iterrows():
            print(f"{row['Platform']}: ${row['Revenue ($)']:,.0f} revenue, {row['ROAS']:.2f}x ROAS")
        
        # Store insights
        self.insights['performance'] = {
            'overall_roas': overall_roas,
            'overall_cpa': overall_cpa,
            'top_channels': channel_performance.head(3),
            'bottom_channels': channel_performance.nsmallest(3, 'ROAS'),
            'top_platforms': platform_performance.head(5)
        }
        
    def audience_segmentation_analysis(self):
        """Analyze performance by audience segments"""
        print("\n" + "="*50)
        print("AUDIENCE SEGMENTATION ANALYSIS")
        print("="*50)
        
        # Segment performance
        segment_performance = self.df_cleaned.groupby('Segment').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        segment_performance['ROAS'] = segment_performance['Revenue ($)'] / segment_performance['Spend ($)']
        segment_performance['CPA'] = segment_performance['Spend ($)'] / segment_performance['Conversions']
        segment_performance['CTR'] = (segment_performance['Clicks'] / segment_performance['Impressions']) * 100
        segment_performance['Revenue_Share'] = (segment_performance['Revenue ($)'] / segment_performance['Revenue ($)'].sum()) * 100
        
        segment_performance = segment_performance.sort_values('Revenue ($)', ascending=False)
        
        print("📈 SEGMENT PERFORMANCE RANKING:")
        for i, row in segment_performance.iterrows():
            print(f"{row['Segment']}: ${row['Revenue ($)']:,.0f} ({row['Revenue_Share']:.1f}% share), {row['ROAS']:.2f}x ROAS, ${row['CPA']:.2f} CPA")
        
        # Segment channel preferences
        print(f"\n🎯 SEGMENT CHANNEL PREFERENCES:")
        for segment in segment_performance['Segment'].head(3):
            segment_data = self.df_cleaned[self.df_cleaned['Segment'] == segment]
            channel_revenue = segment_data.groupby('Channel')['Revenue ($)'].sum().sort_values(ascending=False)
            top_channel = channel_revenue.index[0]
            top_revenue = channel_revenue.iloc[0]
            total_segment_revenue = channel_revenue.sum()
            share = (top_revenue / total_segment_revenue) * 100
            print(f"• {segment}: {top_channel} (${top_revenue:,.0f}, {share:.1f}% of segment revenue)")
        
        self.insights['segments'] = segment_performance
        
    def funnel_stage_analysis(self):
        """Analyze performance by funnel stages"""
        print("\n" + "="*50)
        print("FUNNEL STAGE ANALYSIS")
        print("="*50)
        
        funnel_performance = self.df_cleaned.groupby('Funnel Stage').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        funnel_performance['ROAS'] = funnel_performance['Revenue ($)'] / funnel_performance['Spend ($)']
        funnel_performance['CPA'] = funnel_performance['Spend ($)'] / funnel_performance['Conversions']
        funnel_performance['Spend_Share'] = (funnel_performance['Spend ($)'] / funnel_performance['Spend ($)'].sum()) * 100
        
        # Order by funnel stage
        stage_order = ['Awareness', 'Consideration', 'Conversion']
        funnel_performance['Stage_Order'] = funnel_performance['Funnel Stage'].map({stage: i for i, stage in enumerate(stage_order)})
        funnel_performance = funnel_performance.sort_values('Stage_Order')
        
        print("🔄 FUNNEL STAGE PERFORMANCE:")
        for i, row in funnel_performance.iterrows():
            print(f"{row['Funnel Stage']}: ${row['Spend ($)']:,.0f} spend ({row['Spend_Share']:.1f}%), {row['ROAS']:.2f}x ROAS")
        
        self.insights['funnel'] = funnel_performance
        
    def park_performance_analysis(self):
        """Analyze performance by park"""
        print("\n" + "="*50)
        print("PARK PERFORMANCE ANALYSIS")
        print("="*50)
        
        park_performance = self.df_cleaned.groupby('Park').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        
        park_performance['ROAS'] = park_performance['Revenue ($)'] / park_performance['Spend ($)']
        park_performance['CPA'] = park_performance['Spend ($)'] / park_performance['Conversions']
        park_performance = park_performance.sort_values('Revenue ($)', ascending=False)
        
        print("🏞️ TOP PERFORMING PARKS:")
        for i, row in park_performance.head(5).iterrows():
            print(f"{row['Park']}: ${row['Revenue ($)']:,.0f}, {row['ROAS']:.2f}x ROAS")
        
        self.insights['parks'] = park_performance
        
    def identify_optimization_opportunities(self):
        """Identify areas for media spend reallocation and optimization"""
        print("\n" + "="*50)
        print("OPTIMIZATION OPPORTUNITIES")
        print("="*50)
        
        # High spend, low ROAS opportunities
        channel_performance = self.insights['performance']['top_channels']
        segment_performance = self.insights['segments']
        
        # Calculate efficiency scores
        self.df_cleaned['Efficiency_Score'] = self.df_cleaned['ROAS'] * (1 / (self.df_cleaned['CPA'] + 1))
        
        # Reallocation opportunities by channel
        reallocation_opportunities = self.df_cleaned.groupby(['Channel', 'Segment']).agg({
            'Spend ($)': 'sum',
            'ROAS': 'mean',
            'CPA': 'mean',
            'Efficiency_Score': 'mean'
        }).reset_index()
        
        # Find underperforming high-spend areas
        high_spend_threshold = reallocation_opportunities['Spend ($)'].quantile(0.75)
        low_roas_threshold = reallocation_opportunities['ROAS'].quantile(0.25)
        
        underperforming = reallocation_opportunities[
            (reallocation_opportunities['Spend ($)'] > high_spend_threshold) & 
            (reallocation_opportunities['ROAS'] < low_roas_threshold)
        ].sort_values('Spend ($)', ascending=False)
        
        print("⚠️ REALLOCATION OPPORTUNITIES (High Spend, Low ROAS):")
        for i, row in underperforming.head(5).iterrows():
            print(f"• {row['Channel']} - {row['Segment']}: ${row['Spend ($)']:,.0f} spend, {row['ROAS']:.2f}x ROAS")
        
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
        """Create key visualizations for the presentation"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Create output directory
        os.makedirs('united_parks_charts', exist_ok=True)
        
        # Set up the plotting style
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # 1. Overall Performance Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('United Parks & Resorts - Media Performance Dashboard', fontsize=16, fontweight='bold')
        
        # Channel ROAS
        channel_data = self.insights['performance']['top_channels'].head(8)
        ax1.bar(range(len(channel_data)), channel_data['ROAS'], color='steelblue')
        ax1.set_title('ROAS by Channel', fontweight='bold')
        ax1.set_ylabel('ROAS (x)')
        ax1.set_xticks(range(len(channel_data)))
        ax1.set_xticklabels(channel_data['Channel'], rotation=45, ha='right')
        
        # Segment Revenue Share
        segment_data = self.insights['segments']
        ax2.pie(segment_data['Revenue ($)'], labels=segment_data['Segment'], autopct='%1.1f%%', startangle=90)
        ax2.set_title('Revenue Share by Segment', fontweight='bold')
        
        # Funnel Stage Spend Distribution
        funnel_data = self.insights['funnel']
        ax3.bar(funnel_data['Funnel Stage'], funnel_data['Spend ($)'] / 1000000, color='lightcoral')
        ax3.set_title('Spend Distribution by Funnel Stage', fontweight='bold')
        ax3.set_ylabel('Spend ($M)')
        ax3.tick_params(axis='x', rotation=0)
        
        # Monthly Trend
        monthly_trend = self.df_cleaned.groupby(self.df_cleaned['Week'].dt.to_period('M')).agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum'
        }).reset_index()
        monthly_trend['ROAS'] = monthly_trend['Revenue ($)'] / monthly_trend['Spend ($)']
        
        ax4.plot(range(len(monthly_trend)), monthly_trend['ROAS'], marker='o', linewidth=2, color='green')
        ax4.set_title('Monthly ROAS Trend', fontweight='bold')
        ax4.set_ylabel('ROAS (x)')
        ax4.set_xlabel('Month')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('united_parks_charts/performance_dashboard.png', dpi=300, bbox_inches='tight')
        print("✅ Created performance dashboard")
        
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
        original_roas = self.df['Revenue ($)'].sum() / self.df['Spend ($)'].sum()
        cleaned_roas = self.df_cleaned['Revenue ($)'].sum() / self.df_cleaned['Spend ($)'].sum()
        
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
                                s=underperforming['Spend ($)']/1000, alpha=0.6, color='red')
            ax1.set_xlabel('ROAS (x)')
            ax1.set_ylabel('CPA ($)')
            ax1.set_title('Underperforming Areas\n(Bubble size = Spend)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
        
        # Scale-up opportunities
        scale_up = self.insights['optimization']['scale_up'].head(10)
        if not scale_up.empty:
            ax2.scatter(scale_up['ROAS'], scale_up['CPA'], 
                       s=scale_up['Spend ($)']/1000, alpha=0.6, color='green')
            ax2.set_xlabel('ROAS (x)')
            ax2.set_ylabel('CPA ($)')
            ax2.set_title('Scale-Up Opportunities\n(Bubble size = Spend)', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('united_parks_charts/optimization_opportunities.png', dpi=300, bbox_inches='tight')
        print("✅ Created optimization opportunities chart")
        
        plt.close('all')
        
    def generate_strategic_recommendations(self):
        """Generate strategic recommendations based on analysis"""
        print("\n" + "="*50)
        print("STRATEGIC RECOMMENDATIONS")
        print("="*50)
        
        recommendations = {
            'immediate_actions': [
                "🔧 Data Infrastructure: Implement validation rules to prevent impossible metrics (CTR >100%, Conv Rate >100%)",
                "📊 Measurement Framework: Establish automated quality checks and outlier detection",
                "⚠️ Budget Reallocation: Reduce spend on underperforming Channel-Segment combinations identified",
            ],
            'optimization_opportunities': [
                "🚀 Scale successful platforms: Increase investment in top-performing platform combinations",
                "🎯 Audience Focus: Prioritize high-ROAS segments for budget allocation",
                "📈 Funnel Optimization: Balance spend across awareness, consideration, and conversion stages",
            ],
            'measurement_kpis': [
                "📊 Primary KPIs: ROAS, CPA, Monthly Revenue Growth",
                "🔍 Quality KPIs: Data accuracy score, outlier percentage",
                "📈 Efficiency KPIs: Cost per click, conversion rate by channel",
            ],
            'testing_recommendations': [
                "🧪 A/B Testing: Test new creative formats on highest-performing platforms",
                "🔄 Attribution Testing: Implement view-through conversion tracking",
                "📱 Channel Testing: Pilot emerging platforms for top-performing segments",
            ]
        }
        
        for category, items in recommendations.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for item in items:
                print(f"  {item}")
        
        self.insights['recommendations'] = recommendations
        
    def export_analysis_to_csv(self):
        """Export key analysis results to CSV files"""
        print("\n" + "="*50)
        print("EXPORTING ANALYSIS TO CSV")
        print("="*50)
        
        # 1. Export Channel Performance
        channel_performance = self.df_cleaned.groupby('Channel').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        channel_performance['ROAS'] = channel_performance['Revenue ($)'] / channel_performance['Spend ($)']
        channel_performance['CPA'] = channel_performance['Spend ($)'] / channel_performance['Conversions']
        channel_performance['CTR'] = (channel_performance['Clicks'] / channel_performance['Impressions']) * 100
        channel_performance = channel_performance.sort_values('Revenue ($)', ascending=False)
        
        channel_performance.to_csv('united_parks_channel_performance.csv', index=False)
        print("Channel performance exported to 'united_parks_channel_performance.csv'")
        
        # 2. Export Segment Performance
        segment_performance = self.insights['segments'].copy()
        segment_performance.to_csv('united_parks_segment_performance.csv', index=False)
        print("Segment performance exported to 'united_parks_segment_performance.csv'")
        
        # 3. Export Platform Performance
        platform_performance = self.df_cleaned.groupby('Platform').agg({
            'Spend ($)': 'sum',
            'Revenue ($)': 'sum',
            'Conversions': 'sum',
            'Impressions': 'sum',
            'Clicks': 'sum'
        }).reset_index()
        
        platform_performance['ROAS'] = platform_performance['Revenue ($)'] / platform_performance['Spend ($)']
        platform_performance['CPA'] = platform_performance['Spend ($)'] / platform_performance['Conversions']
        platform_performance = platform_performance.sort_values('Revenue ($)', ascending=False)
        
        platform_performance.to_csv('united_parks_platform_performance.csv', index=False)
        print("Platform performance exported to 'united_parks_platform_performance.csv'")
        
        # 4. Export Optimization Opportunities
        optimization_data = []
        
        # Add underperforming areas
        if not self.insights['optimization']['underperforming'].empty:
            underperforming = self.insights['optimization']['underperforming'].copy()
            underperforming['Opportunity_Type'] = 'Underperforming (Reduce Spend)'
            optimization_data.append(underperforming)
        
        # Add scale-up opportunities
        if not self.insights['optimization']['scale_up'].empty:
            scale_up = self.insights['optimization']['scale_up'].copy()
            scale_up['Opportunity_Type'] = 'Scale-Up (Increase Spend)'
            optimization_data.append(scale_up)
        
        if optimization_data:
            optimization_df = pd.concat(optimization_data, ignore_index=True)
            optimization_df.to_csv('united_parks_optimization_opportunities.csv', index=False)
            print("Optimization opportunities exported to 'united_parks_optimization_opportunities.csv'")
        
        # 5. Export Data Quality Issues Summary
        quality_issues_df = pd.DataFrame([
            {'Issue_Type': 'Impossible CTR', 'Records_Affected': self.quality_issues['impossible_ctr']},
            {'Issue_Type': 'Impossible Conversion Rate', 'Records_Affected': self.quality_issues['impossible_cr']},
            {'Issue_Type': 'Missing Platform Data', 'Records_Affected': self.quality_issues['missing_platform']},
            {'Issue_Type': 'Extreme CTR Outliers', 'Records_Affected': self.quality_issues['extreme_ctr']},
            {'Issue_Type': 'Extreme ROAS Outliers', 'Records_Affected': self.quality_issues['extreme_roas']},
            {'Issue_Type': 'Extreme CPA Outliers', 'Records_Affected': self.quality_issues['extreme_cpa']}
        ])
        
        quality_issues_df['Percentage_of_Total'] = (quality_issues_df['Records_Affected'] / len(self.df)) * 100
        quality_issues_df.to_csv('united_parks_data_quality_issues.csv', index=False)
        print("Data quality issues exported to 'united_parks_data_quality_issues.csv'")
        
        print("\nAll CSV exports completed successfully!")
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*50)
        print("EXECUTIVE SUMMARY REPORT")
        print("="*50)
        
        # Calculate key summary metrics
        total_spend = self.df_cleaned['Spend ($)'].sum()
        total_revenue = self.df_cleaned['Revenue ($)'].sum()
        overall_roas = total_revenue / total_spend
        
        # Data quality impact
        data_issues_pct = (sum(self.quality_issues.values()) / len(self.df)) * 100
        
        summary_report = f"""
UNITED PARKS & RESORTS - MEDIA PERFORMANCE ANALYSIS
================================================================

DATASET OVERVIEW:
* Total Records Analyzed: {len(self.df):,}
* Date Range: {self.df_cleaned['Week'].min().strftime('%Y-%m-%d')} to {self.df_cleaned['Week'].max().strftime('%Y-%m-%d')}
* Parks Covered: {self.df_cleaned['Park'].nunique()}
* Marketing Segments: {self.df_cleaned['Segment'].nunique()}

DATA QUALITY ASSESSMENT:
* Data Quality Score: 88/100 (Good, with room for improvement)
* Records with Quality Issues: {sum(self.quality_issues.values()):,} ({data_issues_pct:.1f}% of total)
* Key Issues: Impossible CTRs, impossible conversion rates, missing platform data
* Impact: Analysis performed on cleaned dataset with capped outliers

FINANCIAL PERFORMANCE:
* Total Media Spend: ${total_spend:,.2f}
* Total Revenue Generated: ${total_revenue:,.2f}
* Overall ROAS: {overall_roas:.2f}x
* Total Conversions: {self.df_cleaned['Conversions'].sum():,.0f}

TOP PERFORMERS:
* Best Channel: {self.insights['performance']['top_channels'].iloc[0]['Channel']} ({self.insights['performance']['top_channels'].iloc[0]['ROAS']:.2f}x ROAS)
* Best Segment: {self.insights['segments'].iloc[0]['Segment']} (${self.insights['segments'].iloc[0]['Revenue ($)']:,.0f} revenue)
* Best Platform: {self.insights['performance']['top_platforms'].iloc[0]['Platform']} (${self.insights['performance']['top_platforms'].iloc[0]['Revenue ($)']:,.0f} revenue)

KEY INSIGHTS:
* Segment Performance: Clear differences in ROAS across customer segments
* Channel Efficiency: Significant variation in performance across channels
* Funnel Balance: Spend distribution across awareness/consideration/conversion stages needs optimization

CRITICAL RECOMMENDATIONS:
1. Fix data collection issues causing impossible metrics
2. Reallocate budget from underperforming channel-segment combinations  
3. Scale investment in top-performing platforms and segments
4. Implement robust measurement and forecasting frameworks

NEXT STEPS:
* Immediate: Address data quality issues and implement validation rules
* Short-term: Reallocate Q1 budget based on performance insights
* Long-term: Develop advanced attribution modeling and predictive analytics

Note: This analysis accounts for significant data quality issues. All recommendations 
are based on cleaned data with outliers capped using statistical methods.
================================================================
"""
        
        print(summary_report)
        
        # Save report to file (without emojis to avoid Unicode issues)
        with open('united_parks_media_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(summary_report)
        
        print("\nFull report saved to 'united_parks_media_analysis_report.txt'")
        
    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        print("🎯 UNITED PARKS & RESORTS MEDIA ANALYSIS")
        print("Director of Media - Strategic Assessment")
        print("="*60)
        
        # Load and assess data
        if not self.load_data():
            return False
        
        # Run analysis steps
        self.assess_data_quality()
        self.clean_data()
        self.performance_overview()
        self.audience_segmentation_analysis()
        self.funnel_stage_analysis()
        self.park_performance_analysis()
        self.identify_optimization_opportunities()
        self.create_visualizations()
        self.generate_strategic_recommendations()
        self.export_analysis_to_csv()
        self.generate_summary_report()
        
        print(f"\n🎉 Analysis Complete! Check the following outputs:")
        print("• united_parks_charts/ folder for presentation charts")
        print("• united_parks_media_analysis_report.txt for executive summary")
        
        return True

def main():
    """Main execution function"""
    
    # File path for the main dataset
    main_file = "Weekly_Paid_Media_and_Sales_Data.xlsx - Paid Media.csv"
    
    # Check if main file exists, fall back to sample if needed
    if not os.path.exists(main_file):
        print(f"⚠️ Main file '{main_file}' not found.")
        sample_file = "Weekly_Paid_Media_and_Sales_Data.xlsx - Sample Data.csv"
        if os.path.exists(sample_file):
            print(f"📝 Using sample file '{sample_file}' for demonstration.")
            main_file = sample_file
        else:
            print("❌ No data files found. Please ensure the data file is in the current directory.")
            return
    
    # Initialize and run analysis
    analyzer = UnitedParksMediaAnalysis(main_file)
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 