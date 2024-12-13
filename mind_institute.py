import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import pandas as pd
import os
import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
train_df_path = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
train_df = pd.read_csv(train_df_path)
test_df_path = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'
test_df = pd.read_csv(test_df_path)


# Lấy danh sách cột chung giữa hai file
common_columns = set(train_df.columns).intersection(set(test_df.columns))

# Sử dụng các cột chung, loại bỏ 'id' nếu có
common_columns = common_columns - {'id'}
X = train_df[list(common_columns)]
y = train_df['sii']

# Điền giá trị thiếu bằng giá trị phổ biến nhất (mode) hoặc giá trị trung bình (mean)
X = X.fillna(X.mode().iloc[0])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuyển đổi các cột dạng object thành category
for col in X_train.select_dtypes(include=['object']).columns:
    X_train[col] = X_train[col].astype('category')
    X_valid[col] = X_valid[col].astype('category')


# Khởi tạo mô hìnhxgb_model = xgb.XGBClassifier(
    random_state=42,        # Đảm bảo tính tái lập kết quả
    n_estimators=200,       # Số lượng cây tăng cường, tăng từ 100 lên 200
    learning_rate=0.05,     # Giảm tốc độ học để mô hình học chậm hơn nhưng kỹ hơn
    max_depth=6,            # Độ sâu tối đa của mỗi cây, kiểm soát độ phức tạp
    subsample=0.8,          # Tỷ lệ mẫu ngẫu nhiên cho mỗi cây để giảm overfitting
    colsample_bytree=0.8,   # Tỷ lệ cột được chọn để xây dựng mỗi cây
    reg_alpha=1,            # Thêm regularization L1 để làm giảm độ phức tạp của mô hình
    reg_lambda=1,           # Thêm regularization L2 để chống overfitting
    enable_categorical=True # Hỗ trợ xử lý trực tiếp dữ liệu phân loại
)

# Huấn luyện mô hình
xgb_model.fit(X_train, y_train)



# Dự đoán
y_pred = xgb_model.predict(X_valid)

# Xử lý giá trị NaN trong y_valid và y_pred
y_valid = np.nan_to_num(np.array(y_valid), nan=np.nanmean(y_valid))
y_pred = np.nan_to_num(np.array(y_pred), nan=np.nanmean(y_pred))

# Loại bỏ các giá trị không hợp lệ
valid_mask = ~y_train.isna() & ~np.isinf(y_train)
X_train = X_train[valid_mask]
y_train = y_train[valid_mask]

# Tính toán RMSE
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print("RMSE:", rmse)


# Chọn cột chung trong test
X_test = test_df[list(common_columns)]
X_test = X_test.fillna(X.mode().iloc[0])

# Chuyển đổi object sang category nếu cần
for col in X_test.select_dtypes(include=['object']).columns:
    X_test[col] = X_test[col].astype('category')

# Dự đoán trên test
y_test_pred = xgb_model.predict(X_test)
print("Dự đoán trên test:", y_test_pred)
