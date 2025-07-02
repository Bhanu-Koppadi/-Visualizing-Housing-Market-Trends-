import pandas as pd

# =====================================================
# 🗂️ MODULE 1: Dataset Overview
# =====================================================
print("\n📦 [MODULE 1: Dataset Overview]")
df = pd.read_csv('dataset.csv')
print("✅ Dataset loaded successfully!")

print(f"🔢 Total Records   : {df.shape[0]}")
print(f"📐 Total Features  : {df.shape[1]}")
print("\n📋 Column Names:")
print("-" * 50)
for col in df.columns:
    print(f"• {col}")
print("-" * 50)

print("\n🔍 Sample Records (Top 5 Rows):")
print(df.head().to_markdown(index=False))

# =====================================================
# 🧹 MODULE 2: Duplicate Removal
# =====================================================
print("\n🧹 [MODULE 2: Duplicate Removal]")
before = df.shape[0]
df = df.drop_duplicates()
after = df.shape[0]

print(f"📝 Rows Before Deduplication : {before}")
print(f"❌ Duplicates Removed         : {before - after}")
print(f"📊 Rows After Deduplication  : {after}")


# =====================================================
# 🔎 MODULE 3: Missing Value Check
# =====================================================
print("\n🔎 [MODULE 3: Missing Value Check]")
missing = df.isnull().sum()
missing = missing[missing > 0]

if not missing.empty:
    print("⚠️ Missing Values Found:")
    print(missing.to_frame(name='Missing Count').to_markdown())
else:
    print("✅ No Missing Values Detected!")


# =====================================================
# 🚫 MODULE 4: Outlier Detection & Removal
# =====================================================
print("\n🚫 [MODULE 4: Outlier Detection & Removal]")

# Summary of extremes
print(f"🏠 Max Bedrooms        : {df['No of Bedrooms'].max()}")
print(f"🛁 Max Bathrooms       : {df['No of Bathrooms'].max()}")
print(f"📏 Min Flat Area (sqft): {df['Flat Area (in Sqft)'].min()}")

# Logical filtering to remove extreme outliers
df = df[(df['No of Bedrooms'] <= 10) &
        (df['No of Bathrooms'] <= 6) &
        (df['Flat Area (in Sqft)'] >= 300)]

print("\n✅ Outliers Removed Based On Rules:")
print("   • Bedrooms > 10")
print("   • Bathrooms > 6")
print("   • Flat Area < 300 sqft")
print(f"📉 Rows Remaining After Outlier Removal: {df.shape[0]}")


# =====================================================
# 🛠️ MODULE 5: Feature Engineering
# =====================================================
print("\n🛠️ [MODULE 5: Feature Engineering]")

# Create new features
df['Total Area (in Sqft)'] = df['Flat Area (in Sqft)'] + df['Basement Area (in Sqft)']
df['House Age Group'] = pd.cut(
    df['Age of House (in Years)'],
    bins=[0, 10, 20, 30, 40, 100],
    labels=['0-10', '11-20', '21-30', '31-40', '40+']
)

print("🆕 New Features Added:")
print("   • Total Area (in Sqft)")
print("   • House Age Group (Binned Age)")

print("\n🔍 Sample of Engineered Features:")
print(df[['Flat Area (in Sqft)', 'Basement Area (in Sqft)', 'Total Area (in Sqft)',
          'Age of House (in Years)', 'House Age Group']].head().to_markdown(index=False))


# =====================================================
# 💾 MODULE 6: Save Cleaned Dataset
# =====================================================
print("\n💾 [MODULE 6: Save Cleaned Dataset]")
df.to_csv("Cleaned_Dataset.csv", index=False)
print("📁 Cleaned dataset saved as: Cleaned_Dataset.csv")
