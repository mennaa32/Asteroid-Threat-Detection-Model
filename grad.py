
import requests
from datetime import datetime, timedelta


import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns

# Feature Selection & Engineering
from sklearn.preprocessing import MinMaxScaler

# model and trinning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# URL الخاص بالـ API
from datetime import datetime, timedelta

# حل مشكله ان الداتا الخطر قليله
from imblearn.over_sampling import SMOTE

# ==============================
#        Data Collection
# ==============================


# جلب تاريخ اليوم
today = datetime.now().strftime("%Y-%m-%d")

# جلب تاريخ آخر 5 أيام
five_days_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

# ربط التاريخ بالـ API
url = f"https://api.nasa.gov/neo/rest/v1/feed?start_date={five_days_ago}&end_date={today}&detailed=false&api_key=On1l7e6HF8kTWgZvZ9tjZTfCe4ZOdJ3ZceBNvmgn"

# إرسال الطلب وجلب البيانات
response = requests.get(url)
data = response.json()

# استخراج البيانات من الـ JSON
neos = []
for date, objects in data["near_earth_objects"].items():
    for obj in objects:
        neos.append({
            "id": obj["id"],
            "name": obj["name"],
            "absolute_magnitude_h": obj["absolute_magnitude_h"],
            "estimated_diameter_min": obj["estimated_diameter"]["kilometers"]["estimated_diameter_min"],
            "estimated_diameter_max": obj["estimated_diameter"]["kilometers"]["estimated_diameter_max"],
            "is_potentially_hazardous_asteroid": obj["is_potentially_hazardous_asteroid"],
            "relative_velocity": float(obj["close_approach_data"][0]["relative_velocity"]["kilometers_per_hour"]),
            "miss_distance": float(obj["close_approach_data"][0]["miss_distance"]["kilometers"]),
            "orbiting_body": obj["close_approach_data"][0]["orbiting_body"],
            "close_approach_date": obj["close_approach_data"][0]["close_approach_date"]
        })

# تحويل البيانات إلى DataFrame
df = pd.DataFrame(neos)

# تحويل 'close_approach_date' إلى datetime
df['close_approach_date'] = pd.to_datetime(df['close_approach_date'])

# ترتيب البيانات حسب 'close_approach_date' (من الأقدم إلى الأحدث)
df_sorted = df.sort_values(by='close_approach_date', ascending=True)

# معلومات عن البيانات
print(df.info())

# عرض أخر 5 صفوف
print(df_sorted.tail())

# عرض اول 5 صفوف
print(df_sorted.head())


# ==============================
#    Exploratory Data Analysis (EDA)
# ==============================

# 1️⃣ وصف إحصائي للبيانات
print(df.describe())

# 2️⃣ توزيع الكويكبات من حيث الخطورة
sns.countplot(data=df, x='is_potentially_hazardous_asteroid')
plt.title('Distribution of Hazardous Asteroids')
plt.show()


# 4️⃣ رسم بياني للمسافات عن الأرض
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='miss_distance', y='relative_velocity', hue='is_potentially_hazardous_asteroid')
plt.title('Miss Distance vs Relative Velocity')
plt.show()

# 5️⃣ عرض القيم الفارغة إن وجدت
print("\nMissing Values:\n", df.isnull().sum())

# 6️⃣ عرض الـ Duplicates إن وجدت
print("\nDuplicated Rows:\n", df.duplicated().sum())

# ==============================
#       Data Cleaning
# ==============================

# 1️⃣ إزالة الـ Duplicates إن وجدت
df.drop_duplicates(inplace=True)

# 2️⃣ إزالة الـ Missing Values إن وجدت
df.dropna(inplace=True)

# 3️⃣ التأكد من الـ Data Types وتعديلها إذا لزم الأمر
print("\nData Types Before:\n", df.dtypes)

# تحويل العمود 'close_approach_date' إلى نوع تاريخ
df['close_approach_date'] = pd.to_datetime(df['close_approach_date'])

print("\nData Types After:\n", df.dtypes)

# 4️⃣ إعادة الفحص للتأكد أن كل شيء تمام
print("\nData after cleaning:\n")
print(df.info())
print(df.head())

# ==============================
# Feature Selection & Engineering
# ==============================


# اختيار الـ Features المهمة فقط
features = df[['absolute_magnitude_h', 'estimated_diameter_min', 'estimated_diameter_max',
               'relative_velocity', 'miss_distance']]

# تطبيع البيانات (Normalization)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# تحويل البيانات مرة تانية إلى DataFrame
features_df = pd.DataFrame(features_scaled, columns=features.columns)

# عرض البيانات بعد التطبيع
print("\nFeatures after Scaling:\n")
print(features_df.head())

# إضافة الـ Target Column
features_df['is_potentially_hazardous_asteroid'] = df['is_potentially_hazardous_asteroid']

# عرض الشكل النهائي للبيانات
print("\nFinal Dataset Ready for Modeling:\n")
print(features_df.head())


# ==============================
#       Model Selection & Training
# ==============================


# 1️⃣ تقسيم البيانات إلى Training و Testing
X = features_df.drop('is_potentially_hazardous_asteroid', axis=1)
y = features_df['is_potentially_hazardous_asteroid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2️⃣ اختيار النموذج (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3️⃣ التنبؤ بالنتائج
y_pred = model.predict(X_test)

# 4️⃣ تقييم النموذج
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5️⃣ Confusion Matrix
# plt.figure(figsize=(5, 4))
# sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
#             xticklabels=['Not Hazardous', 'Hazardous'],
#             yticklabels=['Not Hazardous', 'Hazardous'])
# plt.title('Confusion Matrix')
# plt.show()

plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred, labels=[False, True])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Hazardous', 'Hazardous'],
            yticklabels=['Not Hazardous', 'Hazardous'])
plt.title('Confusion Matrix')
plt.show()


# ==============================
#       Feature Importance
# ==============================
plt.figure(figsize=(8, 5))
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(5).plot(kind='barh', color='skyblue')
plt.title('Feature Importances')
plt.show()




# ==============================
#       Handling Imbalanced Data
# ==============================

# تطبيق SMOTE لزيادة العينات
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts()}")
print(f"After SMOTE: {y_resampled.value_counts()}")

# 1️⃣ تقسيم البيانات من جديد بعد التوازن
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# 2️⃣ تدريب النموذج (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3️⃣ التنبؤ بالنتائج
y_pred = model.predict(X_test)

# 4️⃣ تقييم النموذج
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 5️⃣ رسم الـ Confusion Matrix
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, y_pred, labels=[False, True])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Hazardous', 'Hazardous'],
            yticklabels=['Not Hazardous', 'Hazardous'])
plt.title('Confusion Matrix')
plt.show()




# ==============================
#       Manual Prediction
# ==============================

print("\n🔍 أدخل بيانات الكويكب يدويًا:")

# أخذ القيم من المستخدم
absolute_magnitude_h = float(input("Absolute Magnitude : "))
estimated_diameter_min = float(input("Estimated Diameter Min (km): "))
estimated_diameter_max = float(input("Estimated Diameter Max (km): "))
relative_velocity = float(input("Relative Velocity (km\h): "))
miss_distance = float(input("Miss Distance (km): "))

# بناء DataFrame بالبيانات
manual_input = pd.DataFrame([[
    absolute_magnitude_h,
    estimated_diameter_min,
    estimated_diameter_max,
    relative_velocity,
    miss_distance
]], columns=X.columns)

# تطبيع البيانات زي ما عملنا في التدريب
manual_input_scaled = scaler.transform(manual_input)

# التنبؤ
manual_pred = model.predict(manual_input_scaled)

# طباعة النتيجة
print("\n⚠️ result:")
if manual_pred[0]:
    print("🚨danger")
else:
    print("✅ safe")
