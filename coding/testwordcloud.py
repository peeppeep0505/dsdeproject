import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import re



df = pd.read_csv('final.csv')
df2 = pd.read_csv('finalscrap.csv')
selected_df2 = df2[["title", "Year", "citedby"]].rename(columns={"Year": "year"})
selected_df = df[["title", "Year", "citedby"]].rename(columns={"Year": "year"})
combined_df = pd.concat([selected_df, selected_df2], ignore_index=True)
combined_df = combined_df.drop_duplicates()
# 1. Clean data: แปลง 'title' ให้เป็นข้อความที่มีแต่คำภาษาอังกฤษ
combined_df['title'] = combined_df['title'].apply(lambda x: ' '.join([word for word in x.split() if re.match('^[a-zA-Z]+$', word)]))

# 2. แปลงข้อมูลปี (year) เป็นค่าตัวเลขโดยใช้การแปลงค่า 2018-2024
year_mapping = {2018: 1, 2019: 2, 2020: 3, 2021: 4, 2022: 5, 2023: 6, 2024: 7}
combined_df['year'] = combined_df['year'].map(year_mapping)

# 3. ใช้ TF-IDF เพื่อแปลงคำใน 'title' เป็นเวกเตอร์
vectorizer = TfidfVectorizer(stop_words='english')  # ใช้ stop_words='english' เพื่อกำจัดคำที่ไม่มีความสำคัญ
X_title = vectorizer.fit_transform(combined_df['title'])

# 4. รวมข้อมูลจาก 'year' กับเวกเตอร์จาก TF-IDF
X_year = np.array(combined_df['year']).reshape(-1, 1)  # แปลง 'year' เป็น array ที่มีมิติ (n_samples, 1)
X = np.hstack((X_title.toarray(), X_year))  # รวม X_title กับ X_year

# 5. กำหนด 'citedby' เป็น target
y = combined_df['citedby']

# 6. แบ่งข้อมูลเป็น train และ test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. สร้างโมเดลและฝึกข้อมูล
model = LinearRegression()
model.fit(X_train, y_train)

# 8. ทำนายค่า 'citedby' ใน test set
predicted_citedby = model.predict(X_test)

# 9. ประเมินผลด้วย MSE (Mean Squared Error)
mse = mean_squared_error(y_test, predicted_citedby)
print(f'Mean Squared Error: {mse}')

# 10. แสดงคำที่มีผลกระทบต่อการทำนาย
coefficients = model.coef_[:-1]  # เอาค่า coefficients สำหรับคำออก (เนื่องจากปีไม่ได้เป็นส่วนหนึ่งของคำ)
feature_names = vectorizer.get_feature_names_out()

# สร้าง DataFrame เพื่อแสดงคำและค่าสัมประสิทธิ์
df_coefficients = pd.DataFrame(list(zip(feature_names, coefficients)), columns=['word', 'coefficient'])

# เรียงลำดับคำที่มีค่าสัมประสิทธิ์สูงสุด
df_coefficients = df_coefficients.sort_values(by='coefficient', ascending=False)

# แสดงผลลัพธ์
print(df_coefficients.head(10))  # แสดงคำที่มีค่าสัมประสิทธิ์สูงสุด 10 อันดับแรก
df_coefficients.to_csv('testtf2.csv')