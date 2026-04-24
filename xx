import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==================== Page Config ====================
st.set_page_config(
    page_title="E-Commerce Customer Analytics & RFM Segmentation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A1A2E;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #16213E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .segment-card {
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 0.5rem;
    }
    .insight-box {
        background: #EEF2FF;
        border-left: 4px solid #4338CA;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
        font-size: 0.95rem;
    }
    .kpi-positive { color: #059669; }
    .kpi-negative { color: #DC2626; }
</style>
""", unsafe_allow_html=True)


# ==================== Data Generation ====================
@st.cache_data
def generate_ecommerce_data():
    np.random.seed(123)
    n_customers = 800
    ref_date = datetime(2024, 12, 31)

    categories = {
        'Electronics': (['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch',
                          'Camera', 'Speaker', 'Monitor'], (50, 1500)),
        'Clothing': (['T-Shirt', 'Jeans', 'Jacket', 'Dress', 'Sneakers',
                       'Hoodie', 'Sweater', 'Shorts'], (15, 200)),
        'Home & Kitchen': (['Coffee Maker', 'Blender', 'Cookware Set', 'Vacuum',
                             'Air Purifier', 'Lamp', 'Pillow Set', 'Towel Set'], (20, 300)),
        'Books': (['Fiction Novel', 'Self-Help Book', 'Textbook', 'Comic Book',
                    'Cookbook', 'Biography', 'Science Book', 'Art Book'], (8, 50)),
        'Sports': (['Yoga Mat', 'Dumbbell Set', 'Running Shoes', 'Tennis Racket',
                     'Bicycle', 'Fitness Tracker', 'Gym Bag', 'Water Bottle'], (10, 500)),
        'Beauty': (['Skincare Set', 'Perfume', 'Makeup Kit', 'Hair Dryer',
                     'Sunscreen', 'Moisturizer', 'Lipstick Set', 'Face Mask Pack'], (10, 150)),
    }

    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Google Pay']
    countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada',
                 'Australia', 'Japan', 'Brazil', 'India', 'Singapore']
    country_weights = [0.30, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.06, 0.07, 0.05]

    customer_profiles = []
    for i in range(n_customers):
        profile_type = np.random.choice(['champion', 'loyal', 'potential', 'at_risk', 'lost'],
                                         p=[0.08, 0.15, 0.25, 0.20, 0.32])
        if profile_type == 'champion':
            freq = np.random.randint(15, 40)
            recency_days = np.random.randint(1, 30)
        elif profile_type == 'loyal':
            freq = np.random.randint(8, 20)
            recency_days = np.random.randint(10, 60)
        elif profile_type == 'potential':
            freq = np.random.randint(3, 10)
            recency_days = np.random.randint(20, 120)
        elif profile_type == 'at_risk':
            freq = np.random.randint(5, 15)
            recency_days = np.random.randint(90, 250)
        else:
            freq = np.random.randint(1, 5)
            recency_days = np.random.randint(200, 700)

        customer_profiles.append({
            'customer_id': f'CUST_{i+1:04d}',
            'profile_type': profile_type,
            'order_count': freq,
            'recency_days': recency_days,
            'country': np.random.choice(countries, p=country_weights),
            'preferred_payment': np.random.choice(payment_methods, p=[0.35, 0.20, 0.25, 0.10, 0.10]),
        })

    rows = []
    order_id = 1000
    for cp in customer_profiles:
        for j in range(cp['order_count']):
            order_id += 1
            days_ago = cp['recency_days'] + np.random.randint(0, 600)
            order_date = ref_date - timedelta(days=int(min(days_ago, 729)))

            cat = np.random.choice(list(categories.keys()),
                                    p=[0.20, 0.22, 0.15, 0.13, 0.15, 0.15])
            products, (low, high) = categories[cat]
            product = np.random.choice(products)
            quantity = np.random.choice([1, 1, 1, 2, 2, 3])
            unit_price = round(np.random.uniform(low, high), 2)
            total = round(unit_price * quantity, 2)

            rows.append({
                'order_id': f'ORD-{order_id}',
                'customer_id': cp['customer_id'],
                'order_date': order_date.strftime('%Y-%m-%d'),
                'product_name': product,
                'category': cat,
                'quantity': quantity,
                'unit_price': unit_price,
                'total_amount': total,
                'payment_method': cp['preferred_payment'] if np.random.random() > 0.2
                                  else np.random.choice(payment_methods),
                'country': cp['country'],
            })

    df = pd.DataFrame(rows)
    df['order_date'] = pd.to_datetime(df['order_date'])

    dup_idx = np.random.choice(len(df), 15, replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    nan_idx = np.random.choice(len(df), 20, replace=False)
    df.loc[nan_idx[:10], 'payment_method'] = np.nan
    df.loc[nan_idx[10:], 'country'] = np.nan

    neg_idx = np.random.choice(len(df), 8, replace=False)
    df.loc[neg_idx, 'quantity'] = -1
    df.loc[neg_idx, 'total_amount'] = -df.loc[neg_idx, 'unit_price']

    return df


@st.cache_data
def clean_data(df):
    df_clean = df.copy()

    df_clean = df_clean.drop_duplicates(subset=['order_id', 'customer_id', 'order_date', 'product_name'])

    df_clean = df_clean[df_clean['quantity'] > 0]

    df_clean['payment_method'] = df_clean['payment_method'].fillna('Unknown')
    df_clean['country'] = df_clean['country'].fillna('Unknown')

    df_clean['total_amount'] = df_clean['unit_price'] * df_clean['quantity']

    df_clean['order_month'] = df_clean['order_date'].dt.to_period('M').astype(str)
    df_clean['order_dow'] = df_clean['order_date'].dt.day_name()
    rng = np.random.RandomState(42)
    df_clean['order_hour'] = rng.randint(8, 23, len(df_clean))

    return df_clean


@st.cache_data
def compute_rfm(df):
    ref_date = df['order_date'].max() + timedelta(days=1)

    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (ref_date - x.max()).days,
        'order_id': 'nunique',
        'total_amount': 'sum'
    }).reset_index()
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

    rfm['r_score'] = pd.qcut(rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['rfm_score'] = rfm['r_score'] * 100 + rfm['f_score'] * 10 + rfm['m_score']

    def segment_customer(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r >= 3 and f >= 1 and m >= 3:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost'
        else:
            return 'Need Attention'

    rfm['segment'] = rfm.apply(segment_customer, axis=1)
    return rfm


@st.cache_data
def run_kmeans(rfm, n_clusters=4):
    features = rfm[['recency', 'frequency', 'monetary']].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm_clustered = rfm.copy()
    rfm_clustered['cluster'] = kmeans.fit_predict(scaled)

    cluster_summary = rfm_clustered.groupby('cluster').agg({
        'recency': 'mean', 'frequency': 'mean', 'monetary': ['mean', 'sum'],
        'customer_id': 'count'
    }).round(1)
    cluster_summary.columns = ['Avg Recency', 'Avg Frequency', 'Avg Monetary', 'Total Revenue', 'Count']

    return rfm_clustered, cluster_summary


# ==================== Load & Process ====================
raw_df = generate_ecommerce_data()
df = clean_data(raw_df)
rfm = compute_rfm(df)

# ==================== Sidebar ====================
with st.sidebar:
    st.markdown("## 🎛️ Analytics Filters")
    st.markdown("---")

    date_range = st.date_input(
        "Date Range",
        value=(df['order_date'].min().date(), df['order_date'].max().date()),
        min_value=df['order_date'].min().date(),
        max_value=df['order_date'].max().date(),
    )

    selected_categories = st.multiselect(
        "Product Categories",
        options=sorted(df['category'].unique()),
        default=sorted(df['category'].unique()),
    )

    selected_countries = st.multiselect(
        "Countries (leave empty for all)",
        options=sorted(df[df['country'] != 'Unknown']['country'].unique()),
        default=[],
    )

    n_clusters = st.slider("K-Means Clusters", 3, 6, 4)

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown(
        "This tool analyzes e-commerce transaction data using RFM "
        "(Recency, Frequency, Monetary) segmentation and K-Means "
        "clustering to identify customer groups."
    )

# ==================== Filter ====================
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_date, end_date = date_range[0], date_range[1]
else:
    start_date = end_date = date_range if not isinstance(date_range, (list, tuple)) else date_range[0]

filtered = df[
    (df['order_date'].dt.date >= start_date) &
    (df['order_date'].dt.date <= end_date) &
    (df['category'].isin(selected_categories))
]
if selected_countries:
    filtered = filtered[filtered['country'].isin(selected_countries)]

if filtered.empty:
    st.warning("No data matches the current filters. Please adjust your selections.")
    st.stop()

rfm_filtered = compute_rfm(filtered)
rfm_clustered, cluster_summary = run_kmeans(rfm_filtered, n_clusters)

# ==================== Header ====================
st.markdown('<div class="main-header">🛒 E-Commerce Customer Analytics & RFM Segmentation</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Discover customer segments, purchase patterns, and revenue drivers with data-driven insights'
    '</div>',
    unsafe_allow_html=True
)

# ==================== KPI Cards ====================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Revenue", f"${filtered['total_amount'].sum():,.0f}")
with col2:
    st.metric("Total Orders", f"{filtered['order_id'].nunique():,}")
with col3:
    st.metric("Unique Customers", f"{filtered['customer_id'].nunique():,}")
with col4:
    aov = filtered['total_amount'].mean()
    st.metric("Avg Order Value", f"${aov:,.2f}")
with col5:
    repeat_rate = (rfm_filtered['frequency'] > 1).mean() * 100
    st.metric("Repeat Rate", f"{repeat_rate:.1f}%")

st.markdown("---")

# ==================== Tabs ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Sales Overview", "🎯 RFM Segmentation", "🤖 K-Means Clustering",
    "🛍️ Product Analysis", "👤 Customer Lookup", "📋 Raw Data"
])

# ==================== Tab 1: Sales Overview ====================
with tab1:
    st.subheader("Sales Performance Overview")

    col_rev, col_orders = st.columns(2)

    with col_rev:
        monthly = filtered.groupby(filtered['order_date'].dt.to_period('M')).agg(
            revenue=('total_amount', 'sum'),
            orders=('order_id', 'nunique')
        ).reset_index()
        monthly['order_date'] = monthly['order_date'].astype(str)

        fig_rev = px.area(
            monthly, x='order_date', y='revenue',
            title='Monthly Revenue Trend',
            labels={'order_date': 'Month', 'revenue': 'Revenue ($)'},
            color_discrete_sequence=['#4338CA'],
        )
        fig_rev.update_layout(height=400)
        st.plotly_chart(fig_rev, width='stretch')

    with col_orders:
        fig_ord = px.bar(
            monthly, x='order_date', y='orders',
            title='Monthly Order Count',
            labels={'order_date': 'Month', 'orders': 'Orders'},
            color_discrete_sequence=['#7C3AED'],
        )
        fig_ord.update_layout(height=400)
        st.plotly_chart(fig_ord, width='stretch')

    col_cat, col_pay = st.columns(2)

    with col_cat:
        cat_rev = filtered.groupby('category')['total_amount'].sum().sort_values(ascending=False).reset_index()
        fig_cat = px.bar(
            cat_rev, x='category', y='total_amount',
            title='Revenue by Category',
            labels={'category': 'Category', 'total_amount': 'Revenue ($)'},
            color='total_amount', color_continuous_scale='Purples',
        )
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, width='stretch')

    with col_pay:
        pay_dist = filtered['payment_method'].value_counts().reset_index()
        pay_dist.columns = ['Payment Method', 'Count']
        fig_pay = px.pie(
            pay_dist, values='Count', names='Payment Method',
            title='Payment Method Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pay.update_layout(height=400)
        st.plotly_chart(fig_pay, width='stretch')

    top_rev = cat_rev.iloc[0]['category']
    st.markdown(
        f'<div class="insight-box">'
        f'<strong>Sales Insight:</strong> {top_rev} is the top revenue-generating category. '
        f'The repeat purchase rate of {repeat_rate:.1f}% indicates room for improvement in '
        f'customer retention strategies. Credit Card and PayPal dominate payment preferences.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ==================== Tab 2: RFM Segmentation ====================
with tab2:
    st.subheader("RFM Customer Segmentation")
    st.markdown(
        "**RFM Analysis** segments customers based on:\n"
        "- **Recency**: How recently they purchased\n"
        "- **Frequency**: How often they purchase\n"
        "- **Monetary**: How much they spend"
    )

    segment_counts = rfm_filtered['segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']

    segment_colors = {
        'Champions': '#059669', 'Loyal Customers': '#10B981',
        'Potential Loyalists': '#6366F1', 'New Customers': '#3B82F6',
        'Need Attention': '#F59E0B', 'At Risk': '#EF4444',
        'Cannot Lose': '#DC2626', 'Lost': '#6B7280',
    }

    col_seg, col_tree = st.columns([1, 1])

    with col_seg:
        fig_seg = px.bar(
            segment_counts.sort_values('Count', ascending=True),
            x='Count', y='Segment', orientation='h',
            title='Customer Count by Segment',
            color='Segment', color_discrete_map=segment_colors,
        )
        fig_seg.update_layout(height=450, showlegend=False)
        st.plotly_chart(fig_seg, width='stretch')

    with col_tree:
        seg_revenue = rfm_filtered.groupby('segment')['monetary'].sum().reset_index()
        seg_revenue.columns = ['Segment', 'Total Revenue']
        fig_tree = px.treemap(
            seg_revenue, path=['Segment'], values='Total Revenue',
            title='Revenue Contribution by Segment',
            color='Total Revenue', color_continuous_scale='Blues',
        )
        fig_tree.update_layout(height=450)
        st.plotly_chart(fig_tree, width='stretch')

    seg_detail = rfm_filtered.groupby('segment').agg({
        'recency': 'mean', 'frequency': 'mean', 'monetary': ['mean', 'sum'],
        'customer_id': 'count'
    }).round(1)
    seg_detail.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)',
                           'Total Revenue ($)', 'Customer Count']
    seg_detail = seg_detail.sort_values('Total Revenue ($)', ascending=False)

    st.markdown("**Segment Profile Details**")
    st.dataframe(seg_detail.style.format({
        'Avg Monetary ($)': '${:,.1f}',
        'Total Revenue ($)': '${:,.0f}',
    }), width='stretch')

    fig_rfm_scatter = px.scatter(
        rfm_filtered, x='recency', y='monetary', size='frequency',
        color='segment', hover_data=['customer_id'],
        title='RFM Scatter Plot (Size = Frequency)',
        labels={'recency': 'Recency (days)', 'monetary': 'Total Spend ($)'},
        color_discrete_map=segment_colors,
    )
    fig_rfm_scatter.update_layout(height=500)
    st.plotly_chart(fig_rfm_scatter, width='stretch')

    champ_pct = (rfm_filtered['segment'] == 'Champions').mean() * 100
    champ_rev = rfm_filtered[rfm_filtered['segment'] == 'Champions']['monetary'].sum()
    total_rev = rfm_filtered['monetary'].sum()
    champ_rev_pct = champ_rev / total_rev * 100 if total_rev > 0 else 0

    st.markdown(
        f'<div class="insight-box">'
        f'<strong>RFM Insight:</strong> Champions represent only {champ_pct:.1f}% of customers but '
        f'contribute {champ_rev_pct:.1f}% of total revenue — a classic Pareto pattern. '
        f'"At Risk" and "Lost" segments present retention opportunities worth investigating.'
        f'</div>',
        unsafe_allow_html=True,
    )

# ==================== Tab 3: K-Means Clustering ====================
with tab3:
    st.subheader("K-Means Customer Clustering")
    st.markdown(
        "Machine learning-based segmentation using K-Means on standardized RFM features."
    )

    col_3d, col_summary = st.columns([2, 1])

    with col_3d:
        fig_3d = px.scatter_3d(
            rfm_clustered, x='recency', y='frequency', z='monetary',
            color='cluster', symbol='cluster',
            title=f'3D K-Means Clusters (K={n_clusters})',
            labels={'recency': 'Recency', 'frequency': 'Frequency', 'monetary': 'Monetary ($)'},
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.7,
        )
        fig_3d.update_layout(height=550)
        st.plotly_chart(fig_3d, width='stretch')

    with col_summary:
        st.markdown("**Cluster Profiles**")
        st.dataframe(cluster_summary.style.format({
            'Avg Recency': '{:.0f} days',
            'Avg Frequency': '{:.1f}',
            'Avg Monetary': '${:,.0f}',
            'Total Revenue': '${:,.0f}',
        }), width='stretch')

        inertias = []
        K_range = range(2, 9)
        for k in K_range:
            features = rfm_filtered[['recency', 'frequency', 'monetary']]
            scaler = StandardScaler()
            scaled = scaler.fit_transform(features)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(scaled)
            inertias.append(km.inertia_)

        fig_elbow = px.line(
            x=list(K_range), y=inertias,
            title='Elbow Method (Optimal K)',
            labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
            markers=True,
        )
        fig_elbow.update_layout(height=300)
        st.plotly_chart(fig_elbow, width='stretch')

    col_box1, col_box2, col_box3 = st.columns(3)
    with col_box1:
        fig_b1 = px.box(rfm_clustered, x='cluster', y='recency', color='cluster',
                         title='Recency by Cluster', color_discrete_sequence=px.colors.qualitative.Bold)
        fig_b1.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_b1, width='stretch')
    with col_box2:
        fig_b2 = px.box(rfm_clustered, x='cluster', y='frequency', color='cluster',
                         title='Frequency by Cluster', color_discrete_sequence=px.colors.qualitative.Bold)
        fig_b2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_b2, width='stretch')
    with col_box3:
        fig_b3 = px.box(rfm_clustered, x='cluster', y='monetary', color='cluster',
                         title='Monetary by Cluster', color_discrete_sequence=px.colors.qualitative.Bold)
        fig_b3.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_b3, width='stretch')

    st.markdown(
        '<div class="insight-box">'
        '<strong>Clustering Insight:</strong> K-Means provides a data-driven alternative to '
        'rule-based RFM segmentation. The Elbow Method suggests an optimal K around 4, '
        'where clusters typically represent: high-value active, moderate regulars, '
        'occasional buyers, and dormant/lost customers.'
        '</div>',
        unsafe_allow_html=True,
    )

# ==================== Tab 4: Product Analysis ====================
with tab4:
    st.subheader("Product & Category Analysis")

    col_top, col_cross = st.columns(2)

    with col_top:
        top_products = filtered.groupby('product_name').agg(
            revenue=('total_amount', 'sum'),
            orders=('order_id', 'count'),
            avg_price=('unit_price', 'mean')
        ).sort_values('revenue', ascending=False).head(15).reset_index()

        fig_prod = px.bar(
            top_products, x='revenue', y='product_name', orientation='h',
            title='Top 15 Products by Revenue',
            labels={'revenue': 'Revenue ($)', 'product_name': 'Product'},
            color='revenue', color_continuous_scale='Viridis',
        )
        fig_prod.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_prod, width='stretch')

    with col_cross:
        cat_month = filtered.groupby([
            filtered['order_date'].dt.to_period('M').astype(str), 'category'
        ])['total_amount'].sum().reset_index()
        cat_month.columns = ['Month', 'Category', 'Revenue']

        fig_stack = px.area(
            cat_month, x='Month', y='Revenue', color='Category',
            title='Category Revenue Trends Over Time',
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_stack.update_layout(height=500)
        st.plotly_chart(fig_stack, width='stretch')

    col_heatmap, col_basket = st.columns(2)

    with col_heatmap:
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_cat = filtered.groupby(['order_dow', 'category'])['total_amount'].sum().reset_index()
        dow_pivot = dow_cat.pivot(index='order_dow', columns='category', values='total_amount').fillna(0)
        dow_pivot = dow_pivot.reindex([d for d in dow_order if d in dow_pivot.index])

        fig_heat = px.imshow(
            dow_pivot.values,
            x=dow_pivot.columns.tolist(),
            y=dow_pivot.index.tolist(),
            title='Revenue Heatmap: Day of Week × Category',
            color_continuous_scale='YlOrRd',
            labels=dict(color='Revenue ($)'),
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, width='stretch')

    with col_basket:
        basket = filtered.groupby('category').agg(
            avg_quantity=('quantity', 'mean'),
            avg_price=('unit_price', 'mean'),
            avg_order_value=('total_amount', 'mean'),
        ).round(2).reset_index()

        fig_basket = px.scatter(
            basket, x='avg_price', y='avg_quantity',
            size='avg_order_value', color='category',
            title='Category Analysis: Price vs Quantity (Size = AOV)',
            labels={'avg_price': 'Avg Unit Price ($)', 'avg_quantity': 'Avg Quantity per Order'},
        )
        fig_basket.update_layout(height=400)
        st.plotly_chart(fig_basket, width='stretch')

    st.markdown(
        '<div class="insight-box">'
        '<strong>Product Insight:</strong> Electronics drive the highest per-order revenue but '
        'lower volume. Clothing and Beauty have higher purchase frequency with lower basket '
        'values. Weekend sales patterns vary by category, suggesting targeted promotion timing.'
        '</div>',
        unsafe_allow_html=True,
    )

# ==================== Tab 5: Customer Lookup ====================
with tab5:
    st.subheader("Individual Customer Profile")

    customer_list = sorted(filtered['customer_id'].unique())
    selected_customer = st.selectbox("Select Customer", customer_list)

    if selected_customer:
        cust_orders = filtered[filtered['customer_id'] == selected_customer].sort_values('order_date')
        cust_rfm = rfm_filtered[rfm_filtered['customer_id'] == selected_customer]

        col_info, col_metrics = st.columns([1, 2])

        with col_info:
            st.markdown("#### Customer Info")
            if not cust_rfm.empty:
                st.markdown(f"**Segment:** {cust_rfm.iloc[0]['segment']}")
                st.markdown(f"**Recency:** {cust_rfm.iloc[0]['recency']} days")
                st.markdown(f"**Frequency:** {cust_rfm.iloc[0]['frequency']} orders")
                st.markdown(f"**Monetary:** ${cust_rfm.iloc[0]['monetary']:,.2f}")
                st.markdown(f"**RFM Score:** {cust_rfm.iloc[0]['rfm_score']}")
                st.markdown(f"**Country:** {cust_orders.iloc[0]['country']}")
                st.markdown(f"**Preferred Payment:** {cust_orders['payment_method'].mode().iloc[0]}")

        with col_metrics:
            fig_cust = px.scatter(
                cust_orders, x='order_date', y='total_amount',
                size='quantity', color='category',
                title=f'Purchase History - {selected_customer}',
                labels={'order_date': 'Date', 'total_amount': 'Order Amount ($)'},
            )
            fig_cust.update_layout(height=350)
            st.plotly_chart(fig_cust, width='stretch')

        st.markdown("**Order History**")
        st.dataframe(
            cust_orders[['order_date', 'product_name', 'category', 'quantity',
                          'unit_price', 'total_amount', 'payment_method']].style.format({
                'unit_price': '${:,.2f}', 'total_amount': '${:,.2f}'
            }),
            width='stretch',
        )

# ==================== Tab 6: Raw Data ====================
with tab6:
    st.subheader("Data Explorer")

    col_stats, col_clean = st.columns(2)
    with col_stats:
        st.markdown("**Dataset Statistics**")
        numeric_desc = filtered.select_dtypes(include='number').describe().round(2)
        st.dataframe(numeric_desc, width='stretch')
    with col_clean:
        st.markdown("**Data Cleaning Summary**")
        st.markdown(f"- Raw records: **{len(raw_df):,}**")
        st.markdown(f"- After cleaning: **{len(df):,}**")
        st.markdown(f"- Duplicates removed: **{len(raw_df) - len(raw_df.drop_duplicates(subset=['order_id','customer_id','order_date','product_name'])):,}**")
        st.markdown(f"- Negative quantities removed: **{(raw_df['quantity'] < 0).sum()}**")
        st.markdown(f"- Missing values filled: **{raw_df.isnull().sum().sum()}**")

    st.markdown("**Browse Filtered Data**")
    st.dataframe(filtered, width='stretch', height=400)

    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Data (CSV)", csv,
                       "ecommerce_filtered.csv", "text/csv")

    csv_rfm = rfm_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download RFM Analysis (CSV)", csv_rfm,
                       "rfm_analysis.csv", "text/csv")

# ==================== Footer ====================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.85rem;'>"
    "ACC102 Python Data Product | E-Commerce Customer Analytics & RFM Segmentation<br>"
    "Synthetic e-commerce data | Built with Streamlit, Plotly & Scikit-learn"
    "</div>",
    unsafe_allow_html=True,
)
