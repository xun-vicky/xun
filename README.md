🛒 E-Commerce Customer Analytics & RFM Segmentation Tool

An interactive Streamlit application that analyzes e-commerce transaction data using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment customers and uncover actionable business insights.

Problem & Users

Research Question: How can we segment e-commerce customers to identify high-value groups, at-risk customers, and opportunities for targeted marketing?

Target Users: Marketing managers, CRM analysts, and business strategists who need data-driven customer segmentation to design retention campaigns, optimize promotional spend, and maximize customer lifetime value.

Data

Source: Synthetic e-commerce transaction dataset modeling a multi-category online retailer
Generation Date: Programmatically generated with realistic purchase patterns
Scope: 800 customers, ~7,000 transactions, 6 product categories, 10 countries, 2 years (2023-2024)
Key Fields:
order_id, order_date — Transaction metadata
customer_id, country — Customer identifiers
product_name, category — Product taxonomy
quantity, unit_price, total_amount — Purchase metrics
payment_method — Payment channel
Method (Python Core Steps)

Data Generation — Synthetic data with realistic customer behavior profiles (champions, loyals, at-risk, lost), including deliberate quality issues
Data Cleaning — Duplicate removal, negative quantity filtering, missing value handling, feature engineering (order month, day of week)
RFM Analysis — Recency/Frequency/Monetary computation, quintile scoring (1-5), rule-based customer segmentation into 8 segments
K-Means Clustering — StandardScaler normalization, Elbow Method for optimal K, unsupervised clustering on RFM features
Interactive Visualization — Sales dashboards, segment treemaps, 3D cluster plots, customer lookup, product heatmaps
Main Findings

Pareto Effect: Champions (~8% of customers) generate ~35% of total revenue, confirming the 80/20 principle
At-Risk Opportunity: ~20% of customers are "At Risk" — previously active but recently absent, representing high-value retention targets
Category-Value Link: Electronics drives highest per-order revenue; Clothing has the highest frequency — different strategies needed per category
K-Means Validates RFM: Data-driven clusters align with rule-based segments, strengthening confidence in the segmentation approach
Repeat Rate Gap: Repeat purchase rate below industry benchmarks suggests untapped loyalty program potential
How to Run
