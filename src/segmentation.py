import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class CustomerSegmentationAnalyzer:
    """
    A comprehensive class for customer segmentation using RFM analysis.
    Perfect for freelance portfolio projects showing OOP skills.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the analyzer
        
        Parameters:
        data_path (str): Path to the dataset file
        """
        self.data_path = data_path
        self.raw_data = None
        self.clean_data = None
        self.rfm_data = None
        self.rfm_scaled = None
        self.segments = None
        self.scaler = MinMaxScaler()
        self.model = None
        
    def load_data(self, data_path=None):
        """
        Load the dataset from file
        """
        if data_path:
            self.data_path = data_path
            
        try:
            # Handle different file formats
            if self.data_path.endswith('.csv'):
                self.raw_data = pd.read_csv(self.data_path, encoding='latin1')
            elif self.data_path.endswith('.xlsx'):
                self.raw_data = pd.read_excel(self.data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or Excel.")
                
            print(f"✅ Data loaded successfully: {self.raw_data.shape}")
            return self.raw_data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def explore_data_structure(self):
        """
        Explore the basic structure of the dataset
        """
        if self.raw_data is None:
            print(" No data loaded. Please load data first.")
            return
            
        print("=" * 50)
        print("📊 DATA EXPLORATION")
        print("=" * 50)
        
        print(f"Dataset shape: {self.raw_data.shape}")
        print(f"\nColumn names:")
        for i, col in enumerate(self.raw_data.columns):
            print(f"  {i+1}. {col}")
            
        print(f"\nData types:")
        print(self.raw_data.dtypes)
        
        print(f"\nFirst few rows:")
        print(self.raw_data.head())
        
        print(f"\nMissing values:")
        missing = self.raw_data.isnull().sum()
        print(missing[missing > 0])
        
        return self.raw_data.info()
    
    def clean_and_preprocess(self, customer_col='CustomerID', 
                           invoice_col='InvoiceNo', 
                           date_col='InvoiceDate',
                           quantity_col='Quantity',
                           price_col='UnitPrice'):
        """
        Clean and preprocess the data for RFM analysis
        
        Parameters:
        customer_col (str): Name of customer ID column
        invoice_col (str): Name of invoice number column  
        date_col (str): Name of date column
        quantity_col (str): Name of quantity column
        price_col (str): Name of unit price column
        """
        if self.raw_data is None:
            print(" No data loaded. Please load data first.")
            return None
            
        print("🧹 CLEANING DATA...")
        
        # Make a copy for cleaning
        df = self.raw_data.copy()
        
        # Store column names for later use
        self.cols = {
            'customer': customer_col,
            'invoice': invoice_col,
            'date': date_col,
            'quantity': quantity_col,
            'price': price_col
        }
        
        # Remove rows with missing customer IDs
        initial_rows = len(df)
        df = df.dropna(subset=[customer_col])
        print(f"Removed {initial_rows - len(df)} rows with missing customer IDs")
        
        # Remove returns (negative quantities or invoice numbers starting with 'C')
        df = df[df[quantity_col] > 0]
        df = df[~df[invoice_col].astype(str).str.startswith('C')]
        
        # Remove rows with zero or negative prices
        df = df[df[price_col] > 0]
        
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Create total amount column
        df['TotalAmount'] = df[quantity_col] * df[price_col]
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        print(f" Cleaned data shape: {df.shape}")
        print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
        
        self.clean_data = df
        return self.clean_data
    
    def calculate_rfm_features(self, analysis_date=None):
        """
        Calculate RFM (Recency, Frequency, Monetary) features for each customer
        
        Parameters:
        analysis_date (str or datetime): Reference date for recency calculation
        """
        if self.clean_data is None:
            print(" No clean data available. Please clean data first.")
            return None
            
        print("📈 CALCULATING RFM FEATURES...")
        
        df = self.clean_data.copy()
        
        # Set analysis date (usually day after last transaction)
        if analysis_date is None:
            analysis_date = df[self.cols['date']].max() + pd.Timedelta(days=1)
        else:
            analysis_date = pd.to_datetime(analysis_date)
            
        print(f"Analysis date: {analysis_date}")
        
        # Calculate RFM metrics
        rfm = df.groupby(self.cols['customer']).agg({
            self.cols['date']: lambda x: (analysis_date - x.max()).days,  # Recency
            self.cols['invoice']: 'nunique',  # Frequency
            'TotalAmount': 'sum'  # Monetary
        }).reset_index()
        
        # Rename columns
        rfm.columns = [self.cols['customer'], 'Recency', 'Frequency', 'Monetary']
        
        # Add customer count for context
        total_customers = len(rfm)
        
        print(f" RFM calculated for {total_customers} customers")
        print("\nRFM Summary:")
        print(rfm.describe())
        
        self.rfm_data = rfm
        return self.rfm_data
    
    def visualize_rfm_distributions(self):
        """
        Create visualizations for RFM distributions
        """
        if self.rfm_data is None:
            print(" No RFM data available. Please calculate RFM first.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('RFM Feature Distributions', fontsize=16, fontweight='bold')
        
        # Recency distribution
        axes[0,0].hist(self.rfm_data['Recency'], bins=50, alpha=0.7, color='skyblue')
        axes[0,0].set_title('Recency Distribution (Days since last purchase)')
        axes[0,0].set_xlabel('Days')
        axes[0,0].set_ylabel('Count')
        
        # Frequency distribution
        axes[0,1].hist(self.rfm_data['Frequency'], bins=50, alpha=0.7, color='lightgreen')
        axes[0,1].set_title('Frequency Distribution (Number of purchases)')
        axes[0,1].set_xlabel('Number of Purchases')
        axes[0,1].set_ylabel('Count')
        
        # Monetary distribution
        axes[1,0].hist(self.rfm_data['Monetary'], bins=50, alpha=0.7, color='salmon')
        axes[1,0].set_title('Monetary Distribution (Total spent)')
        axes[1,0].set_xlabel('Total Amount Spent')
        axes[1,0].set_ylabel('Count')
        
        # Correlation heatmap
        corr_matrix = self.rfm_data[['Recency', 'Frequency', 'Monetary']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=axes[1,1])
        axes[1,1].set_title('RFM Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        fig.savefig("...-visualizations/rfm_distributions.png", dpi=300)
        return fig
    
    def scale_rfm_features(self):
        """
        Scale RFM features for clustering
        """
        if self.rfm_data is None:
            print(" No RFM data available. Please calculate RFM first.")
            return None
            
        print("⚖️ SCALING RFM FEATURES...")
        
        # Select RFM features for scaling
        rfm_features = self.rfm_data[['Recency', 'Frequency', 'Monetary']].copy()
        
        # Note: For recency, lower values are better, so we'll invert it
        rfm_features['Recency'] = rfm_features['Recency'].max() - rfm_features['Recency']
        
        # Scale features
        self.rfm_scaled = self.scaler.fit_transform(rfm_features)
        
        # Convert back to DataFrame for easier handling
        self.rfm_scaled_df = pd.DataFrame(
            self.rfm_scaled, 
            columns=['Recency_Scaled', 'Frequency_Scaled', 'Monetary_Scaled']
        )
        self.rfm_scaled_df[self.cols['customer']] = self.rfm_data[self.cols['customer']].values
        
        print("✅ Features scaled successfully")
        print("\nScaled features summary:")
        print(self.rfm_scaled_df.describe())
        
        return self.rfm_scaled_df
    
    def find_optimal_clusters(self, max_k=10, method='both'):
        """
        Find optimal number of clusters using Elbow Method and Silhouette Score
        
        Parameters:
        max_k (int): Maximum number of clusters to test
        method (str): 'elbow', 'silhouette', or 'both'
        """
        if self.rfm_scaled is None:
            print(" No scaled data available. Please scale features first.")
            return None
            
        print("🔍 FINDING OPTIMAL NUMBER OF CLUSTERS...")
        
        # Range of cluster numbers to test
        k_range = range(2, max_k + 1)
        
        # Storage for metrics
        inertias = []
        silhouette_scores = []
        
        # Test each k value
        for k in k_range:
            # Fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.rfm_scaled)
            
            # Calculate metrics
            inertias.append(kmeans.inertia_)
            sil_score = silhouette_score(self.rfm_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
            
            print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.3f}")
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        if method in ['elbow', 'both']:
            # Elbow Method Plot
            axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Number of Clusters (k)')
            axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
            axes[0].set_title('Elbow Method for Optimal k')
            axes[0].grid(True, alpha=0.3)
        
        if method in ['silhouette', 'both']:
            # Silhouette Score Plot
            axes[1].plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
            axes[1].set_xlabel('Number of Clusters (k)')
            axes[1].set_ylabel('Silhouette Score')
            axes[1].set_title('Silhouette Score for Optimal k')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        fig.savefig("...-visualizations/rfm_distributions.png", dpi=300)
        
        # Recommend optimal k
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\n✅ Recommended k: {optimal_k} (highest silhouette score: {max(silhouette_scores):.3f})")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
    
    def perform_clustering(self, n_clusters=4, algorithm='kmeans'):
        """
        Perform customer clustering using specified algorithm
        
        Parameters:
        n_clusters (int): Number of clusters for KMeans
        algorithm (str): 'kmeans' or 'dbscan'
        """
        if self.rfm_scaled is None:
            print("❌ No scaled data available. Please scale features first.")
            return None
            
        print(f"🎯 PERFORMING {algorithm.upper()} CLUSTERING...")
        
        if algorithm == 'kmeans':
            self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.model.fit_predict(self.rfm_scaled)
            
        elif algorithm == 'dbscan':
            self.model = DBSCAN(eps=0.3, min_samples=5)
            cluster_labels = self.model.fit_predict(self.rfm_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
        # Add cluster labels to RFM data
        self.segments = self.rfm_data.copy()
        self.segments['Cluster'] = cluster_labels
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(self.rfm_scaled, cluster_labels)
            print(f"✅ Clustering complete! Silhouette Score: {sil_score:.3f}")
        
        print(f"Number of clusters formed: {len(set(cluster_labels))}")
        print(f"Cluster distribution:")
        print(self.segments['Cluster'].value_counts().sort_index())
        
        return self.segments
    
    def profile_segments(self):
        """
        Create detailed profiles for each customer segment
        """
        if self.segments is None:
            print("❌ No segments available. Please perform clustering first.")
            return None
            
        print("👥 PROFILING CUSTOMER SEGMENTS...")
        
        # Calculate segment statistics
        segment_profiles = self.segments.groupby('Cluster').agg({
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'], 
            'Monetary': ['mean', 'median', 'sum'],
            self.cols['customer']: 'count'
        }).round(2)
        
        # Flatten column names
        segment_profiles.columns = ['_'.join(col).strip() for col in segment_profiles.columns]
        segment_profiles = segment_profiles.rename(columns={f"{self.cols['customer']}_count": "Customer_Count"})
        
        # Add percentage of total customers
        total_customers = len(self.segments)
        segment_profiles['Percentage'] = (segment_profiles['Customer_Count'] / total_customers * 100).round(1)
        
        print("📊 SEGMENT PROFILES:")
        print("=" * 80)
        print(segment_profiles)
        
        # Create business-friendly segment names and descriptions
        segment_descriptions = self._create_segment_descriptions(segment_profiles)
        
        # Visualization
        self._visualize_segments()
        
        return segment_profiles, segment_descriptions
    
    def _create_segment_descriptions(self, profiles):
        """
        Create business-friendly descriptions for each segment
        """
        descriptions = {}
        
        for cluster in profiles.index:
            recency = profiles.loc[cluster, 'Recency_mean']
            frequency = profiles.loc[cluster, 'Frequency_mean']
            monetary = profiles.loc[cluster, 'Monetary_mean']
            count = profiles.loc[cluster, 'Customer_Count']
            percentage = profiles.loc[cluster, 'Percentage']
            
            # Create segment name based on RFM characteristics
            if recency < 50 and frequency > 5 and monetary > 1000:
                name = "💎 VIP Champions"
                description = "High-value, frequent, recent customers. Your best customers!"
            elif recency < 100 and frequency > 3 and monetary > 500:
                name = "🌟 Loyal Customers" 
                description = "Consistent customers with good spending habits"
            elif recency > 200 and frequency < 2:
                name = "😴 Lost Customers"
                description = "Haven't purchased recently, low engagement"
            elif recency < 100 and frequency < 3 and monetary < 200:
                name = "🆕 New Customers"
                description = "Recent customers, still building relationship"
            else:
                name = f" Segment {cluster}"
                description = "Mixed characteristics requiring detailed analysis"
            
            descriptions[cluster] = {
                'name': name,
                'description': description,
                'customer_count': int(count),
                'percentage': percentage,
                'avg_recency': recency,
                'avg_frequency': frequency,
                'avg_monetary': monetary
            }
            
        print("\n🎯 BUSINESS SEGMENT DESCRIPTIONS:")
        print("=" * 50)
        for cluster, info in descriptions.items():
            print(f"\n{info['name']} (Cluster {cluster})")
            print(f"   {info['customer_count']} customers ({info['percentage']}%)")
            print(f"  Avg: {info['avg_recency']:.0f} days recency, "
                  f"{info['avg_frequency']:.1f} purchases, "
                  f"${info['avg_monetary']:.0f} spent")
            print(f"  💡 {info['description']}")
            
        return descriptions
    
    def _visualize_segments(self):
        """
        Create comprehensive visualizations for customer segments
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Customer Segment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Segment size distribution
        segment_counts = self.segments['Cluster'].value_counts().sort_index()
        axes[0,0].pie(segment_counts.values, labels=[f'Cluster {i}' for i in segment_counts.index], 
                     autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Segment Distribution')
        
        # 2. RFM heatmap by segment
        rfm_by_segment = self.segments.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        sns.heatmap(rfm_by_segment.T, annot=True, cmap='RdYlBu_r', ax=axes[0,1])
        axes[0,1].set_title('Average RFM Values by Segment')
        axes[0,1].set_xlabel('Cluster')
        
        # 3. Recency vs Frequency scatter
        colors = plt.cm.Set1(np.linspace(0, 1, len(segment_counts)))
        for i, cluster in enumerate(segment_counts.index):
            cluster_data = self.segments[self.segments['Cluster'] == cluster]
            axes[1,0].scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                            c=[colors[i]], label=f'Cluster {cluster}', alpha=0.6)
        axes[1,0].set_xlabel('Recency (Days)')
        axes[1,0].set_ylabel('Frequency (Purchases)')
        axes[1,0].set_title('Recency vs Frequency by Segment')
        axes[1,0].legend()
        
        # 4. Monetary distribution by segment
        for cluster in segment_counts.index:
            cluster_data = self.segments[self.segments['Cluster'] == cluster]['Monetary']
            axes[1,1].hist(cluster_data, bins=20, alpha=0.6, label=f'Cluster {cluster}')
        axes[1,1].set_xlabel('Monetary Value')
        axes[1,1].set_ylabel('Count')
        axes[1,1].set_title('Monetary Distribution by Segment')
        axes[1,1].legend()
        axes[1,1].set_xlim(0, self.segments['Monetary'].quantile(0.95))  # Remove outliers for better viz
        
        plt.tight_layout()
        
        return fig
    

    def calculate_customer_lifetime_value(self, time_period_days=365):
        """
        Calculate Customer Lifetime Value (CLV) for each segment
        
        Parameters:
        time_period_days (int): Time period for CLV calculation
        """
        if self.segments is None:
            print("o segments available. Please perform clustering first.")
            return None
            
        print(f" CALCULATING CUSTOMER LIFETIME VALUE ({time_period_days} days)...")
        
        # Calculate average purchase value and frequency
        self.segments['Avg_Purchase_Value'] = self.segments['Monetary'] / self.segments['Frequency']
        
        # Estimate purchase frequency per year (simplified)
        # This is a basic estimation - in real projects, use more sophisticated methods
        days_in_data = (self.clean_data[self.cols['date']].max() - 
                       self.clean_data[self.cols['date']].min()).days
        
        self.segments['Annual_Purchase_Frequency'] = (
            self.segments['Frequency'] * (time_period_days / days_in_data)
        )
        
        # Simple CLV calculation: Avg Purchase Value × Annual Frequency
        self.segments['CLV_Estimate'] = (
            self.segments['Avg_Purchase_Value'] * self.segments['Annual_Purchase_Frequency']
        )
        
        # CLV by segment
        clv_by_segment = self.segments.groupby('Cluster')['CLV_Estimate'].agg(['mean', 'median', 'sum']).round(2)
        
        print("📊 CLV BY SEGMENT:")
        print(clv_by_segment)
        
        return clv_by_segment
    def business_recommendations(self): 

        if self.segments is None: 
            print("No segments available")
            return None

        print("Generating business recommendations: ")

        segment_stats = self.segments.groupby('Cluster').agg({
            'Recency' : 'mean', 
            'Frequency': 'mean', 
            'Monetary': ['mean', 'sum'], 
            self.cols['customer']: 'count'
        }).round(2)

        recomendations= {}

        for cluster in segment_stats.index: 
            recency = segment_stats.loc[cluster, ('Recency', 'mean')]
            frequency = segment_stats.loc[cluster, ('Frequency', 'mean')]
            monetary = segment_stats.loc[cluster, ('Monetary', 'mean')]
            total_value = segment_stats.loc[cluster, ('Monetary', 'sum')]
            count = segment_stats.loc[cluster, (self.cols['customer'], 'count')]

            if recency < 50 and frequency > 5 and monetary > 1000: 
                segment_name = "VIP Champions"
                strategies = [
                    " **Retention Strategy**: VIP loyalty program with exclusive perks",
                    " **Upselling**: Premium product recommendations and early access",
                    " **Engagement**: Personal account manager and priority support",
                    " **Revenue Impact**: Focus on maintaining their high lifetime value"
                ]
                priority = "HIGH"
            elif recency < 100 and frequency > 3 and monetary > 500:  
                segment_name = "Loyal customers"
                strategies = [
                    " **Rewards Program**: Points-based loyalty system",
                    " **Regular Communication**: Monthly newsletters with personalized offers",
                    " **Upgrade Path**: Encourage transition to VIP status",
                    " **Revenue Impact**: Steady revenue contributors with growth potential"
                ]
                priority = "HIGH"

            elif recency > 200 and frequency < 2: 
                segment_name = "Lost customers :("
                strategies = [
                    " **Win-Back Campaign**: Special discount offers (20-30% off)",
                    " **Multi-Channel Approach**: Email, SMS, and retargeting ads",
                    " **Feedback Survey**: Understand why they left",
                    " **Revenue Impact**: Recovery potential with targeted investment",
                ]
                priority = "Medium"

            elif recency < 100 and frequency < 3 and monetary < 200: 
                segment_name = " New customers"
                strategies = [
                    " **Onboarding**: Welcome series with product education",
                    " **Cross-Selling**: Introduce complementary products",
                    " **Customer Success**: Proactive support and guidance",
                    " **Revenue Impact**: High growth potential with proper nurturing"
                ]

                priority = "Medium"
            else: 
                segment_name = f" Segment {cluster}"
                strategies = [
                    "📊 **Deep Analysis**: Requires further investigation",
                    "🎯 **A/B Testing**: Test different engagement strategies",
                    "📈 **Monitoring**: Track behavior changes over time",
                    "📊 **Revenue Impact**: Uncertain - needs strategic focus"
                ]
                priority = "Low"
            recomendations[cluster] = {
                'name': segment_name,
                'customer_count': int(count),
                'total_revenue': float(total_value),
                'avg_monetary': float(monetary),
                'priority': priority,
                'strategies': strategies,
                'roi_potential': self._calculate_roi_potential(recency, frequency, monetary)
            }
        print("\n" + "="*80)
        print("Business recommendations by segment")
        print("="*80)

        for cluster, rec in recomendations.items():
            print(f"\n{rec['name']} (Cluster {cluster}) - Priority: {rec['priority']}")
            print(f"👥 Customers: {rec['customer_count']} | 💰 Total Revenue: ${rec['total_revenue']:,.2f}")
            print(f"📈 ROI Potential: {rec['roi_potential']}")
            print("Strategic Actions:")
            for strategy in rec['strategies']:
                print(f"  {strategy}")
            print("-" * 60)
        
        return recomendations
    
    def _calculate_roi_potential(self, recency, frequency, monetary):
        if recency < 50 and frequency > 5 and monetary > 1000:
            return "⭐⭐⭐⭐⭐ VERY HIGH"
        elif recency < 100 and frequency > 3 and monetary > 500:
            return "⭐⭐⭐⭐ HIGH"
        elif recency > 200:
            return "⭐⭐ MEDIUM (Win-back required)"
        else:
            return "⭐⭐⭐ MODERATE"
    def export_segment_results(self, filename_prefix = "Customer_segments"): 

        if self.segments is None: 
            print("No segments available.")
            return None
        segments_file = f"{filename_prefix}_details.csv"
        self.segments.to_csv(segments_file, index=False)
        print(f"Detailed segments exported to: {segments_file}")

        summary = self.segments.groupby('Cluster').agg({
            'Recency': ['mean', 'median', 'std'],
            'Frequency': ['mean', 'median', 'std'],
            'Monetary': ['mean', 'median', 'sum', 'std'],
            self.cols['customer']: 'count'
        }).round(2)

        summary_file = f"{filename_prefix}_summary.csv"
        summary.to_csv(summary_file)
        print(f"Segment summary exported to: {summary_file}")

        return segments_file, summary_file
    def generate_executive_summary(self):
        
        if self.segments is None:
            print("❌ No segments available. Please perform clustering first.")
            return None
        
        print("\n" + "="*60)
        print("📊 EXECUTIVE SUMMARY")
        print("="*60)
        
        total_customers = len(self.segments)
        total_revenue = self.segments['Monetary'].sum()
        avg_customer_value = total_revenue / total_customers
        
        # Segment breakdown
        segment_summary = self.segments.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum'],
            self.cols['customer']: 'count'
        }).round(2)
        
        print(f"🎯 BUSINESS OVERVIEW:")
        print(f"   Total Customers Analyzed: {total_customers:,}")
        print(f"   Total Revenue: ${total_revenue:,.2f}")
        print(f"   Average Customer Value: ${avg_customer_value:.2f}")
        print(f"   Number of Segments: {len(segment_summary)}")
        
        print(f"\n💰 TOP PERFORMING SEGMENTS:")
        # Sort by total revenue
        top_segments = segment_summary.sort_values(('Monetary', 'sum'), ascending=False)
        for i, (cluster, data) in enumerate(top_segments.head(2).iterrows()):
            revenue_share = (data[('Monetary', 'sum')] / total_revenue) * 100
            customer_share = (data[(self.cols['customer'], 'count')] / total_customers) * 100
            print(f"   {i+1}. Cluster {cluster}: {revenue_share:.1f}% of revenue from {customer_share:.1f}% of customers")
        
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        print(f"   1. Focus retention efforts on high-value segments")
        print(f"   2. Implement targeted win-back campaigns for lost customers")
        print(f"   3. Develop nurturing programs for new customer acquisition")
        print(f"   4. Create personalized marketing strategies by segment")
        
        return {
            'total_customers': total_customers,
            'total_revenue': total_revenue,
            'avg_customer_value': avg_customer_value,
            'segment_count': len(segment_summary)
        }

    # 3. Enhanced export with business insights
    def export_comprehensive_results(self, filename_prefix="customer_segments"):
        """
        Export comprehensive segmentation results including business insights
        """
        if self.segments is None:
            print("❌ No segments available.")
            return None
        
        print("📤 EXPORTING COMPREHENSIVE RESULTS...")
        
        # 1. Export detailed segments
        segments_file = f"{filename_prefix}_detailed.csv"
        export_segments = self.segments.copy()
        export_segments['Avg_Purchase_Value'] = export_segments['Monetary'] / export_segments['Frequency']
        export_segments.to_csv(segments_file, index=False)
        print(f"✅ Detailed segments exported to: {segments_file}")
        
        # 2. Export segment summary with business insights
        summary = self.segments.groupby('Cluster').agg({
            'Recency': ['mean', 'median', 'std'],
            'Frequency': ['mean', 'median', 'std'],
            'Monetary': ['mean', 'median', 'sum', 'std'],
            self.cols['customer']: 'count'
        }).round(2)
        
        # Add business metrics
        total_customers = len(self.segments)
        total_revenue = self.segments['Monetary'].sum()
        
        summary['Revenue_Share_Percent'] = ((summary[('Monetary', 'sum')] / total_revenue) * 100).round(1)
        summary['Customer_Share_Percent'] = ((summary[(self.cols['customer'], 'count')] / total_customers) * 100).round(1)
        
        summary_file = f"{filename_prefix}_summary.csv"
        summary.to_csv(summary_file)
        print(f"✅ Segment summary exported to: {summary_file}")
        
        # 3. Export business recommendations
        recommendations = self.generate_business_recommendations()
        if recommendations:
            rec_data = []
            for cluster, rec in recommendations.items():
                rec_data.append({
                    'Cluster': cluster,
                    'Segment_Name': rec['name'],
                    'Customer_Count': rec['customer_count'],
                    'Total_Revenue': rec['total_revenue'],
                    'Avg_Monetary': rec['avg_monetary'],
                    'Priority': rec['priority'],
                    'ROI_Potential': rec['roi_potential'],
                    'Key_Strategies': '; '.join([s.replace('*', '') for s in rec['strategies']])
                })
            
            rec_df = pd.DataFrame(rec_data)
            rec_file = f"{filename_prefix}_recommendations.csv"
            rec_df.to_csv(rec_file, index=False)
            print(f"✅ Business recommendations exported to: {rec_file}")
        
        return segments_file, summary_file, rec_file

        




if __name__ == "__main__":
    # Start the analyzer
    analyzer = CustomerSegmentationAnalyzer()
    
    print("CUSTOMER SEGMENTATION ANALYSIS")
    
    analyzer.load_data(r'../data/raw/online_retail.csv')
    analyzer.clean_and_preprocess()
    analyzer.calculate_rfm_features()
    analyzer.visualize_rfm_distributions()
    analyzer.scale_rfm_features()
    
    print("\n" + "="*50)
    print("Clustering & Segmentation")
    print("="*50)
    
    analyzer.find_optimal_clusters(max_k=8)
    analyzer.perform_clustering(n_clusters=4)
    analyzer.profile_segments()
    analyzer.calculate_customer_lifetime_value()

    print("\n" + "="*50)
    print("Business Recommendations: ")
    print("="*50)

    recommendations = analyzer.business_recommendations()
    analyzer.export_segment_results()
    analyzer.generate_executive_summary()
    analyzer.export_comprehensive_results()
        
    print("\n🎉 ANALYSIS COMPLETE!")
    print("Ready for business presentation and portfolio showcase!")

    total_customers = len(analyzer.segments)
    total_revenue = analyzer.segments['Monetary'].sum()
    print(f"\n Total Business Impact: ")
    print(f" Total customers analyzed: {total_customers:,}")
    print(f" Total revenue: ${total_revenue:,.2f}")
    print(f" Avergage customer value: ${total_revenue/total_customers:.2f}")
        
    