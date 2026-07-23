import pandas as pd
import numpy as np
import os

# 파일 경로
FILE_PATH = os.path.join("student_habits.csv")


# 1. 데이터 불러오기
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return None

    df = pd.read_csv(file_path, encoding="utf-8-sig")

    print("\n[1. 데이터 불러오기]")
    print("데이터 로드 완료:", df.shape[0], "행 X", df.shape[1], "열")
    return df


# 2. 데이터 구조 확인(explore_structure)
def explore_structure(df):
    print("\n[2. 데이터 구조 확인]")

    print("\n1. 데이터 행, 열:")
    print(df.shape)

    print("\n2. 컬럼, 자료형:")
    print(df.dtypes)

    print("\n3. 상위 5개 데이터:")
    print(df.head())

    print("\n4. 데이터 정보:")
    df.info()


# 3. 기술통계 출력(show_statistics)
def show_statistics(df):
    print("\n[3. 기술통계 출력]")

    # 수치형 컬럼의 기술통계 출력
    print(df.describe().round(3))

    print("\n수치형 컬럼별 평균:")

    number_columns = df.select_dtypes(include=["number"]).columns

    for col in number_columns:      # 수치형 컬럼 각각의 평균을 반복문으로 순회하며 한 줄씩 출력
        print(f"{col}: {round(df[col].mean(), 4)}")
    
    # count : 데이터 수
    # mean : 평균
    # std : 표준편차
    # min : 최소값
    # 25% : 1사분위수
    # 50% : 2사분위수(중앙값)  
    # 75% : 3사분위수
    # max : 최대값

# 4. 결측치 현황 파악(check_missing)
def check_missing(df):
    print("\n[4. 결측치 현황 파악]")

    df_missing = pd.DataFrame({
        "결측치 수": df.isnull().sum(),
        "결측치 비율(%)": df.isnull().mean() * 100
    })

    def missing_ratio(ratio):
        if ratio < 5:
            return "낮음"
        elif ratio < 20:
            return "주의"
        else:
            return "높음"

    df_missing["심각도"] = df_missing["결측치 비율(%)"].apply(missing_ratio)
    df_missing["결측치 비율(%)"] = df_missing["결측치 비율(%)"].round(2)
    print(df_missing)

    return df_missing


# 5. NumPy로 직접 통계량 계산(numpy_stats)
def numpy_stats(df):
    print("\n[5. NumPy 통계량 계산]")

    # 결측치 제거 후 NumPy 배열로 변환(study_hours 컬럼)
    df_null = df.dropna()
    study_hours = df_null.values[:, 5]

    print("\n1. NumPy 기술통계")
    print("mean:", round(np.mean(study_hours), 3)) # 평균
    
    print("std:", round(np.std(study_hours), 3)) # 표준편차

    print("median:", round(np.median(study_hours), 3)) # 중앙값
    print("min:", round(np.min(study_hours), 3)) # 최소값
    print("max:", round(np.max(study_hours), 3)) # 최댓값

    print("\n2. pandas 기술통계")
    pandas_stats = df_null["study_hours"].dropna().agg(["mean", "std", "median", "min", "max"]).round(3)

    print(pandas_stats)

    print("\n3. 6시간 이상 공부하는 학생 수")

    student_count = df_null[df_null["study_hours"] >= 6]

    print(student_count["student_id"].count())


# 6. 전체 함수 연결(main)
def main():
    df = load_data(FILE_PATH)

    if df is not None:
        explore_structure(df)
        show_statistics(df)
        check_missing(df)
        numpy_stats(df)

if __name__ == "__main__":
    main()