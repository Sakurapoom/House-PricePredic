import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1. โหลดข้อมูล 
df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

# 2. 5 Features 
features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
target = 'SalePrice'

X = df[features]
y = df[target]

# จัดการข้อมูลสูญหาย
X = X.fillna(X.median())

# 3. แบ่งข้อมูล Train/Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. สร้างและสอนโมเดล
model = LinearRegression()
model.fit(X_train, y_train)

# 5. ประเมินผล (Evaluate) - เก็บค่าพวกนี้ไว้ใส่ใน Slide นะครับ
y_pred = model.predict(X_test)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.0f}")

# 6. บันทึกโมเดลเป็นไฟล์ .pkl
with open('house_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("โมเดลถูกบันทึกเรียบร้อยในชื่อ 'house_model.pkl'")