# Install dependencies as needed:
# pip install scikit-learn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from math import radians, sin, cos, sqrt, atan2


# ============================================================================
# COMPLETE SOLUTION
# ============================================================================
def top_20_restaurants(df):
    # first graph by category

    sns.set(style='whitegrid')

    top_20_cat = df['cuisine'].value_counts().head(10)

    plt.figure(figsize=(18, 10))
    sns.barplot(x=top_20_cat.values, y=top_20_cat.index, color='steelblue')
    plt.title('Top 20 Restaurant Categories Distribution')
    plt.xlabel('Count')
    plt.ylabel('Category')

    for i, count in enumerate(top_20_cat.values):
        plt.text(count, i, f'{count}', ha='left', va='center')

    plt.tight_layout()
    plt.subplots_adjust(left=0.32)
    plt.show()


def income_corr_graph(df):
    # Clean the data - remove rows with missing values in key columns
    df_clean = df.dropna(subset=['median_income', 'zip_code', 'menu_item'])


    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Income, Zip Code, and Menu Item Analysis', fontsize=16, fontweight='bold')

    # 1. Average income by top menu items
    menu_income = df_clean.groupby('menu_item')['median_income'].agg(['mean', 'count']).reset_index()
    menu_income = menu_income[menu_income['count'] >= 2].sort_values('mean', ascending=False).head(15)

    axes[0, 0].barh(menu_income['menu_item'], menu_income['mean'], color='steelblue')
    axes[0, 0].set_xlabel('Average Median Income ($)', fontsize=11)
    axes[0, 0].set_ylabel('Menu Item', fontsize=11)
    axes[0, 0].set_title('Top 15 Menu Items by Average Area Income', fontsize=12, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)

    # 2. Income distribution by zip code
    zip_income = df_clean.groupby('zip_code')['median_income'].mean().sort_values(ascending=False).head(20)
    axes[0, 1].bar(range(len(zip_income)), zip_income.values, color='coral')
    axes[0, 1].set_xlabel('Zip Code Rank', fontsize=11)
    axes[0, 1].set_ylabel('Median Income ($)', fontsize=11)
    axes[0, 1].set_title('Top 20 Zip Codes by Median Income', fontsize=12, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)

    # 3. Menu item count vs income scatter
    menu_zip_income = df_clean.groupby(['zip_code', 'menu_item']).agg({
        'median_income': 'first',
        'menu_item': 'count'
    }).rename(columns={'menu_item': 'item_count'}).reset_index()

    axes[1, 0].scatter(menu_zip_income['median_income'], menu_zip_income['item_count'],
                       alpha=0.6, s=100, color='green', edgecolors='darkgreen')
    axes[1, 0].set_xlabel('Median Income ($)', fontsize=11)
    axes[1, 0].set_ylabel('Number of Menu Items', fontsize=11)
    axes[1, 0].set_title('Menu Item Variety vs Area Income', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(menu_zip_income['median_income'], menu_zip_income['item_count'], 1)
    p = np.poly1d(z)
    axes[1, 0].plot(menu_zip_income['median_income'], p(menu_zip_income['median_income']),
                    "r--", alpha=0.8, linewidth=2, label=f'Trend line')
    axes[1, 0].legend()

    # 4. Heatmap of top menu items by income brackets
    # Create income brackets
    df_clean['income_bracket'] = pd.cut(df_clean['median_income'],
                                        bins=5,
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # Get top 10 most common menu items
    top_items = df_clean['menu_item'].value_counts().head(10).index
    df_top_items = df_clean[df_clean['menu_item'].isin(top_items)]

    # Create pivot table
    heatmap_data = pd.crosstab(df_top_items['menu_item'], df_top_items['income_bracket'])

    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], cbar_kws={'label': 'Count'})
    axes[1, 1].set_xlabel('Income Bracket', fontsize=11)
    axes[1, 1].set_ylabel('Menu Item', fontsize=11)
    axes[1, 1].set_title('Menu Item Distribution Across Income Brackets', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('income_menu_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print correlation statistics
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    # Calculate correlation between income and menu item price
    if 'price' in df_clean.columns:
        price_income_corr = df_clean[['median_income', 'price']].corr().iloc[0, 1]
        print(f"\nCorrelation between median income and menu item price: {price_income_corr:.3f}")

    # Most common items in high vs low income areas
    high_income = df_clean[df_clean['median_income'] > df_clean['median_income'].quantile(0.75)]
    low_income = df_clean[df_clean['median_income'] < df_clean['median_income'].quantile(0.25)]


    return


def top_clusters(df):
#
    features = ['lat', 'lng', 'score', 'price', 'zip_code']
    cluster_data = df[features].copy()

    # Remove any remaining NaN values
    cluster_data = cluster_data.dropna()

    print("\nFeature statistics:")
    print(cluster_data.describe())

    # Standardize the features (important for clustering)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data)

    # Update data1_cleaned to only include rows that are in cluster_data
    data1_cleaned = df.loc[cluster_data.index]

    # Determine optimal number of clusters using elbow method
    print("\nCalculating optimal number of clusters...")
    inertia_values = []
    K_range = range(2, 11)

    for k in K_range:
        print(f"Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia_values.append(kmeans.inertia_)

    print("Elbow method calculation complete!")

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertia_values, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('elbow_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Perform clustering with optimal k (let's use 5 as a starting point)
    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    data1_cleaned['cluster'] = kmeans.fit_predict(scaled_features)

    print(f"\nClustering with {optimal_k} clusters completed!")
    print("\nCluster distribution:")
    print(df['cluster'].value_counts().sort_index())
    return

def geo_clusters(df):
    # Analyze cluster characteristics


    # Geographic visualization of clusters
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(df['lng'], df['lat'],
                          c=df['cluster'],
                          cmap='viridis',
                          alpha=0.6,
                          s=50,
                          edgecolors='black',
                          linewidth=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geographic Distribution of Restaurant Clusters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cluster_geographic.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Cluster by price range distribution
    plt.figure(figsize=(12, 6))
    cluster_price = pd.crosstab(df['cluster'], data1_cleaned['price_range'], normalize='index') * 100
    cluster_price.plot(kind='bar', stacked=False, color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.title('Price Range Distribution by Cluster')
    plt.legend(title='Price Range')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('cluster_price_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Score distribution by cluster
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='cluster', y='score', data=df, palette='viridis')
    plt.xlabel('Cluster')
    plt.ylabel('Restaurant Score')
    plt.title('Score Distribution by Cluster')
    plt.tight_layout()
    plt.savefig('cluster_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


def haversine_distance(lat1, lon1, lat2, lon2):

    R = 3959  # Earth's radius in miles

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance
def food_desert_analysis(df_clean):
    zip_info = df_clean.groupby('zip_code').agg({
        'zip_lat': 'first',
        'zip_lng': 'first',
        'population': 'first',
        'median_income': 'first'
    }).reset_index()

    # Get unique restaurant locations
    restaurants = df_clean[['restaurant_id', 'lat', 'lng', 'restaurant_name']].drop_duplicates('restaurant_id')

    print(f"\nTotal unique zip codes analyzed: {len(zip_info)}")
    print(f"Total unique restaurants: {len(restaurants)}")

    # Calculate restaurant coverage for each zip code
    DELIVERY_RADIUS = 10  # miles

    coverage_data = []

    for idx, zip_row in zip_info.iterrows():
        zip_code = zip_row['zip_code']
        zip_lat = zip_row['zip_lat']
        zip_lng = zip_row['zip_lng']

        # Count restaurants within delivery radius
        restaurants_in_range = 0
        menu_items_available = 0
        cuisines_available = set()

        for _, rest in restaurants.iterrows():
            distance = haversine_distance(zip_lat, zip_lng, rest['lat'], rest['lng'])

            if distance <= DELIVERY_RADIUS:
                restaurants_in_range += 1

                # Count menu items from this restaurant
                rest_items = df_clean[df_clean['restaurant_id'] == rest['restaurant_id']]
                menu_items_available += len(rest_items)
                cuisines_available.update(rest_items['cuisine'].dropna().unique())

        coverage_data.append({
            'zip_code': zip_code,
            'population': zip_row['population'],
            'median_income': zip_row['median_income'],
            'restaurants_in_range': restaurants_in_range,
            'menu_items_available': menu_items_available,
            'cuisine_variety': len(cuisines_available)
        })

    coverage_df = pd.DataFrame(coverage_data)

    # Define food desert criteria
    # A food desert is typically defined as an area with:
    # 1. Low restaurant access (fewer restaurants within range)
    # 2. Low menu variety
    # 3. Often correlated with lower income or higher population density

    # Calculate thresholds
    low_restaurant_threshold = coverage_df['restaurants_in_range'].quantile(0.25)
    low_variety_threshold = coverage_df['menu_items_available'].quantile(0.25)

    # Identify potential food deserts
    coverage_df['is_food_desert'] = (
            (coverage_df['restaurants_in_range'] <= low_restaurant_threshold) |
            (coverage_df['menu_items_available'] <= low_variety_threshold)
    )

    # Severity levels
    coverage_df['desert_severity'] = 'Well Served'
    coverage_df.loc[coverage_df['restaurants_in_range'] <= low_restaurant_threshold, 'desert_severity'] = 'At Risk'
    coverage_df.loc[
        (coverage_df['restaurants_in_range'] <= low_restaurant_threshold) &
        (coverage_df['menu_items_available'] <= low_variety_threshold),
        'desert_severity'
    ] = 'Food Desert'

    print("\n" + "=" * 70)
    print("FOOD DESERT IDENTIFICATION RESULTS")
    print("=" * 70)

    print(f"\nDelivery radius: {DELIVERY_RADIUS} miles")
    print(f"Low restaurant threshold: ≤{low_restaurant_threshold:.0f} restaurants")
    print(f"Low variety threshold: ≤{low_variety_threshold:.0f} menu items")

    print(f"\n--- Summary Statistics ---")
    print(f"Average restaurants per zip code: {coverage_df['restaurants_in_range'].mean():.1f}")
    print(f"Average menu items available: {coverage_df['menu_items_available'].mean():.1f}")
    print(f"Average cuisine variety: {coverage_df['cuisine_variety'].mean():.1f}")

    print(f"\n--- Food Desert Classification ---")
    print(coverage_df['desert_severity'].value_counts())

    food_deserts = coverage_df[coverage_df['desert_severity'] == 'Food Desert'].sort_values('restaurants_in_range')
    print(f"\n--- IDENTIFIED FOOD DESERTS (Zip Codes) ---")
    if len(food_deserts) > 0:
        for _, row in food_deserts.iterrows():
            print(f"Zip: {int(row['zip_code']):05d} | Restaurants: {row['restaurants_in_range']:.0f} | "
                  f"Menu Items: {row['menu_items_available']:.0f} | Population: {row['population']:.0f} | "
                  f"Income: ${row['median_income']:.0f}")
    else:
        print("No food deserts identified with current criteria.")

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Food Desert Analysis - 10 Mile Delivery Radius', fontsize=16, fontweight='bold')

    # 1. Restaurant coverage by zip code
    colors = {'Well Served': 'green', 'At Risk': 'orange', 'Food Desert': 'red'}
    coverage_sorted = coverage_df.sort_values('restaurants_in_range')

    axes[0, 0].barh(range(len(coverage_sorted)), coverage_sorted['restaurants_in_range'],
                    color=[colors[x] for x in coverage_sorted['desert_severity']])
    axes[0, 0].set_xlabel('Number of Restaurants Within 10 Miles', fontsize=11)
    axes[0, 0].set_ylabel('Zip Code Index', fontsize=11)
    axes[0, 0].set_title('Restaurant Coverage by Zip Code', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(low_restaurant_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='x', alpha=0.3)

    # 2. Income vs Restaurant Access
    scatter_colors = coverage_df['desert_severity'].map(colors)
    axes[0, 1].scatter(coverage_df['median_income'], coverage_df['restaurants_in_range'],
                       c=scatter_colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Median Income ($)', fontsize=11)
    axes[0, 1].set_ylabel('Restaurants Within 10 Miles', fontsize=11)
    axes[0, 1].set_title('Income vs Restaurant Access', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k) for k in ['Well Served', 'At Risk', 'Food Desert']]
    axes[0, 1].legend(handles=legend_elements)

    # 3. Population vs Menu Variety
    axes[1, 0].scatter(coverage_df['population'], coverage_df['menu_items_available'],
                       c=scatter_colors, s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Population', fontsize=11)
    axes[1, 0].set_ylabel('Menu Items Available', fontsize=11)
    axes[1, 0].set_title('Population vs Menu Variety', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(handles=legend_elements)

    # 4. Desert severity distribution
    severity_counts = coverage_df['desert_severity'].value_counts()
    axes[1, 1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%',
                   colors=[colors[x] for x in severity_counts.index], startangle=90,
                   textprops={'fontsize': 11})
    axes[1, 1].set_title('Food Desert Classification Distribution', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('food_desert_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Export results to CSV
    coverage_df.to_csv('food_desert_results.csv', index=False)
    print(f"\n✓ Results exported to 'food_desert_results.csv'")
    print(f"✓ Visualization saved as 'food_desert_analysis.png'")

    # Additional analysis: correlation between income and food access
    correlation = coverage_df[['median_income', 'restaurants_in_range', 'menu_items_available']].corr()
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)
    print(correlation)
    return

# main
data1 = pd.read_csv('./merged_data.csv')

# basic data printout
print(f"Total rows: {len(data1)}")
print(f"Rows after cleaning: {len(data1)}")
print(f"\nMedian Income range: ${data1['median_income'].min():.2f} - ${data1['median_income'].max():.2f}")
print(f"Number of unique zip codes: {data1['zip_code'].nunique()}")
print(f"Number of unique restaurants: {data1['restaurant_name'].nunique()}")
print(f"Number of unique menu items: {data1['menu_item'].nunique()}")


top_20_restaurants(data1)
#income_corr_graph(data1)
#top_clusters(data1)
#geo_clusters(data1)
food_desert_analysis(data1)
print(data1.info())