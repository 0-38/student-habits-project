import pandas as pd
import os
import numpy as np

# 파일 경로
FILE_PATH = os.path.join("week 1","data", "student_habits.csv")

# 데이터 불러오기
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"데이터 로드 실패: {file_path}")
        return None

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    print("데이터 로드 완료:", df.shape[0], "행 X", df.shape[1], "열")
    return df


# 1. 결측치 처리
# sleep_hours, phone_hours, exercise_hours 컬럼의 결측치를 중앙값으로 처리
columns = ['sleep_hours', 'phone_hours', 'exercise_hours']

def handle_missing(df):
    for col in columns:
        df[col] = df[col].fillna(df[col].median())

    is_null = df.isnull().sum().sum()
    
    print("결측치 처리 후 남은 결측치:", is_null, "개")

# 2. 이상치 처리
def handle_outliers(df):
    columns = ['sleep_hours', 'study_hours', 'phone_hours', 'exercise_hours']

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        df[col] = df[col].clip(lower, upper, axis=0)
        
    print("이상치 처리(클리핑) 완료")     
    
# 3. 성별 라벨 인코딩(남: 0 / 여: 1)
def convert_types(df):
    gender_map = {'남': 0, '여': 1}
    df['gender_code'] = df['gender'].map(gender_map)
    
    print("gender → gender_code 인코딩 완료")

# 4. 파생변수 생성 
def add_features(df):
    df['productive_hours'] = df['study_hours'] + df['exercise_hours'] # 하루 생산적 활동 시간
    df['sleep_sufficient'] = (df['sleep_hours'] >= 7).astype(int) # 수면 충족 여부(7시간 이상: 1, 미만: 0)
    df['phone_overuse'] = (df['phone_hours'] >= 4).astype(int) # 스마트폰 과의존 여부(4시간 이상: 1, 미만: 0)

    print("파생변수 생성 완료: productive_hours, sleep_sufficient, phone_overuse")
    
# 5. 전체 연결 + 저장
def main():
    df = load_data(FILE_PATH)
    
    if df is None:
        return None
                
    handle_missing(df)
    handle_outliers(df)
    convert_types(df)
    add_features(df)
    
    return df

if __name__ == "__main__":
    df = main()
    
    df.to_csv("week 2/data/student_habits_clean.csv", index=False, encoding="utf-8-sig")
    print("정제 데이터 저장 완료: week 2/data/student_habits_clean.csv", df.shape[0], "행 X", df.shape[1], "열")