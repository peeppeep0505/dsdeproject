import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
import re

# อ่านข้อมูล
df = pd.read_csv('final.csv')
df2 = pd.read_csv('finalscrap.csv')
selected_df2 = df2[["title", "Year", "citedby"]].rename(columns={"Year": "year"})
selected_df = df[["title", "Year", "citedby"]].rename(columns={"Year": "year"})
combined_df = pd.concat([selected_df, selected_df2], ignore_index=True)
combined_df = combined_df.drop_duplicates()

# Clean data
combined_df['title'] = combined_df['title'].apply(lambda x: ' '.join([word for word in x.split() if re.match('^[a-zA-Z]+$', word)]))
year_mapping = {2018: 1, 2019: 2, 2020: 3, 2021: 4, 2022: 5, 2023: 6, 2024: 7}
combined_df['year'] = combined_df['year'].map(year_mapping)

# ใช้ TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, max_df=0.8, min_df=5)
X_title = vectorizer.fit_transform(combined_df['title'])

# รวมข้อมูลปี
X_year = np.array(combined_df['year']).reshape(-1, 1)
X = hstack((X_title, X_year))
y = combined_df['citedby']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train โมเดล
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# ดึงค่าคำและ coefficient
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[:-1]  # ตัดส่วนของปีออก

# ปรับค่า coefficient ให้ไม่มีค่าติดลบ
coefficients = np.maximum(0, coefficients)

# สร้าง DataFrame สำหรับคำและ citedby (predict)
df_coefficients = pd.DataFrame({'word': feature_names, 'citedby_predict': coefficients})

# Export เป็น CSV
df_coefficients.to_csv('word_citedby_predict.csv', index=False)

print("Exported to 'word_citedby_predict.csv'")
