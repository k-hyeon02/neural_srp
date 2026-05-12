import numpy as np
from utils import generate_regular_polygon 


DICIT_ARRAY_SETUP = {
    'array_type': 'planar',  # 평면 배열 유형 지정
    'mic_pos': np.array([ 
        [ 0.96, 0.00, 0.00],  # DICIT 1번 마이크 좌표
        [ 0.64, 0.00, 0.00],  # DICIT 2번 마이크 좌표
        [ 0.32, 0.00, 0.00],  # DICIT 3번 마이크 좌표
        [ 0.16, 0.00, 0.00],  # DICIT 4번 마이크 좌표
        [ 0.08, 0.00, 0.00],  # DICIT 5번 마이크 좌표
        [ 0.04, 0.00, 0.00],  # DICIT 6번 마이크 좌표
        [ 0.00, 0.00, 0.00],  # DICIT 7번 마이크 좌표
        [ 0.96, 0.00, 0.32],  # DICIT 8번 마이크 좌표
        [-0.04, 0.00, 0.00],  # DICIT 9번 마이크 좌표
        [-0.08, 0.00, 0.00],  # DICIT 10번 마이크 좌표
        [-0.16, 0.00, 0.00],  # DICIT 11번 마이크 좌표
        [-0.32, 0.00, 0.00],  # DICIT 12번 마이크 좌표
        [-0.64, 0.00, 0.00],  # DICIT 13번 마이크 좌표
        [-0.96, 0.00, 0.00],  # DICIT 14번 마이크 좌표
        [-0.96, 0.00, 0.32]  # DICIT 15번 마이크 좌표
    ]),
}

DUMMY_ARRAY_SETUP = {  
    'array_type': 'planar',  # 평면 배열 유형 지정
    'mic_pos': np.array([  
        [-0.079,  0.000, 0.000],  # 더미 1번 마이크 좌표
        [-0.079, -0.009, 0.000],  # 더미 2번 마이크 좌표
        [ 0.079,  0.000, 0.000],  # 더미 3번 마이크 좌표
        [ 0.079, -0.009, 0.000]  # 더미 4번 마이크 좌표
    ]),  
}  

BENCHMARK2_ARRAY_SETUP = {
    'array_type': '3D',  # 3차원 배열 유형 지정
    'mic_pos': np.array([  
        [-0.028,  0.030, -0.040],  # Benchmark2 1번 마이크 좌표
        [ 0.006,  0.057,  0.000],  # Benchmark2 2번 마이크 좌표
        [ 0.022,  0.022, -0.046],  # Benchmark2 3번 마이크 좌표
        [-0.055, -0.024, -0.025],  # Benchmark2 4번 마이크 좌표
        [-0.031,  0.023,  0.042],  # Benchmark2 5번 마이크 좌표
        [-0.032,  0.011,  0.046],  # Benchmark2 6번 마이크 좌표
        [-0.025, -0.003,  0.051],  # Benchmark2 7번 마이크 좌표
        [-0.036, -0.027,  0.038],  # Benchmark2 8번 마이크 좌표
        [-0.035, -0.043,  0.025],  # Benchmark2 9번 마이크 좌표
        [ 0.029, -0.048, -0.012],  # Benchmark2 10번 마이크 좌표
        [ 0.034, -0.030,  0.037],  # Benchmark2 11번 마이크 좌표
        [ 0.035,  0.025,  0.039]  # Benchmark2 12번 마이크 좌표
    ]), 
} 

EIGENMIKE_ARRAY_SETUP = {
    'array_type': '3D',  # 3차원 배열 유형 지정
    'mic_pos': np.array([  
        [ 0.000,  0.039,  0.015],  # Eigenmike 1번 마이크 좌표
        [-0.022,  0.036,  0.000],  # Eigenmike 2번 마이크 좌표
        [ 0.000,  0.039, -0.015],  # Eigenmike 3번 마이크 좌표
        [ 0.022,  0.036,  0.000],  # Eigenmike 4번 마이크 좌표
        [ 0.000,  0.022,  0.036],  # Eigenmike 5번 마이크 좌표
        [-0.024,  0.024,  0.024],  # Eigenmike 6번 마이크 좌표
        [-0.039,  0.015,  0.000],  # Eigenmike 7번 마이크 좌표
        [-0.024,  0.024,  0.024],  # Eigenmike 8번 마이크 좌표
        [ 0.000,  0.022, -0.036],  # Eigenmike 9번 마이크 좌표
        [ 0.024,  0.024, -0.024],  # Eigenmike 10번 마이크 좌표
        [ 0.039,  0.015,  0.000],  # Eigenmike 11번 마이크 좌표
        [ 0.024,  0.024,  0.024],  # Eigenmike 12번 마이크 좌표
        [-0.015,  0.000,  0.039],  # Eigenmike 13번 마이크 좌표
        [-0.036,  0.000,  0.022],  # Eigenmike 14번 마이크 좌표
        [-0.036,  0.000, -0.022],  # Eigenmike 15번 마이크 좌표
        [-0.015,  0.000, -0.039],  # Eigenmike 16번 마이크 좌표
        [ 0.000, -0.039,  0.015],  # Eigenmike 17번 마이크 좌표
        [ 0.022, -0.036,  0.000],  # Eigenmike 18번 마이크 좌표
        [ 0.000, -0.039, -0.015],  # Eigenmike 19번 마이크 좌표
        [-0.022, -0.036,  0.000],  # Eigenmike 20번 마이크 좌표
        [ 0.000, -0.022,  0.036],  # Eigenmike 21번 마이크 좌표
        [ 0.024, -0.024,  0.024],  # Eigenmike 22번 마이크 좌표
        [ 0.039, -0.015,  0.000],  # Eigenmike 23번 마이크 좌표
        [ 0.024, -0.024, -0.024],  # Eigenmike 24번 마이크 좌표
        [ 0.000, -0.022, -0.036],  # Eigenmike 25번 마이크 좌표
        [-0.024, -0.024, -0.024],  # Eigenmike 26번 마이크 좌표
        [-0.039, -0.015,  0.000],  # Eigenmike 27번 마이크 좌표
        [-0.024, -0.024,  0.024],  # Eigenmike 28번 마이크 좌표
        [ 0.015,  0.000,  0.039],  # Eigenmike 29번 마이크 좌표
        [ 0.036,  0.000,  0.022],  # Eigenmike 30번 마이크 좌표
        [ 0.036,  0.000, -0.022],  # Eigenmike 31번 마이크 좌표
        [ 0.015,  0.000, -0.039]  # Eigenmike 32번 마이크 좌표
    ]),
} 

MINIDSP_ARRAY_SETUP = { 
    'array_type': 'planar',  # 평면 배열 유형 지정
    'mic_pos': np.array([  
        [ 0.0000,  0.0430, 0.000],  # miniDSP 1번 마이크 좌표
        [ 0.0372,  0.0215, 0.000],  # miniDSP 2번 마이크 좌표
        [ 0.0372, -0.0215, 0.000],  # miniDSP 3번 마이크 좌표
        [ 0.0000, -0.0430, 0.000],  # miniDSP 4번 마이크 좌표
        [-0.0372, -0.0215, 0.000],  # miniDSP 5번 마이크 좌표
        [-0.0372,  0.0215, 0.000]  # miniDSP 6번 마이크 좌표
    ]), 
}  # min

TAU_NIGENS_TETRAHEDRAL = {
    'array_type': '3D',  # 3차원 배열 유형 지정
    'mic_pos': np.array([  
        [ 0.0243,  0.0243,  0.024],  # TAU-NIGENS 1번 마이크 좌표
        [ 0.0243, -0.0243, -0.024],  # TAU-NIGENS 2번 마이크 좌표
        [-0.0243,  0.0243, -0.024],  # TAU-NIGENS 3번 마이크 좌표
        [-0.0243, -0.0243,  0.024]  # TAU-NIGENS 4번 마이크 좌표
    ]),
}

ARRAY_SETUPS = {
    "benchmark2": BENCHMARK2_ARRAY_SETUP,
    "dicit": DICIT_ARRAY_SETUP, 
    "dummy": DUMMY_ARRAY_SETUP,
    "eigenmike": EIGENMIKE_ARRAY_SETUP, 
    "mini_dsp": MINIDSP_ARRAY_SETUP,  
    "tau_nigens_tetrahedral": TAU_NIGENS_TETRAHEDRAL
} 


# 랜덤 배열 설정 생성 함수
def generate_random_array_setup(radius_range_in_m, n_mics_range,
                                min_dist_between_mics_in_m=0,  # 마이크 간 최소 거리 기본값 인자
                                mode="spherical"):  # 배열 생성 모드 기본값 인자
    """n_mics개의 마이크를 가지는 랜덤 배열 설정을 생성
    min_radius_in_m과 max_radius_in_m 사이의 반지름을 가지는 구면 위에 생성

    Args:
        radius_range_in_m (tuple of floats): (min_radius_in_m, max_radius_in_m).
        n_mics_range (tuple of ints): (min_n_mics, max_n_mics).
        min_dist_between_mics_in_m (float, optional): 미터 단위 마이크 간 최소 거리. 기본값은 0.
        mode (str, optional): 배열 모드. "spherical" 또는 "poly2d" 중 하나.
            "spherical"은 무작위로 할당된 반지름의 구면 안에 점들을 흩뿌림.
            "poly2d"는 n_mics개의 꼭짓점을 가지는 정규 2차원 다각형을 생성.
            기본값은 "spherical".

    Returns:
        dict: 'array_type'과 'mic_pos' 키를 포함하는 딕셔너리.
    """
    array_type = "3D"  # 기본 3차원 배열 유형
    if mode == "poly2d":  # 평면 다각형 모드 조건
        array_type = "planar"  # 평면 배열 유형 대입

    # 마이크 개수 난수 생성 단계
    n_mics = np.random.randint(  # 마이크 개수 난수 생성 호출
        n_mics_range[0], n_mics_range[1]  # 마이크 개수 하한 및 상한 인자
    )  # 마이크 개수 난수 생성 종료

    array_setup = {  
        'array_type': array_type,  # 배열 유형 값 저장
        'mic_pos': np.zeros((n_mics, 3))  # 마이크 좌표 초기 배열 저장
    }  

    # 반지름 난수 생성 단계
    radius_in_m = np.random.uniform(radius_range_in_m[0],  # 반지름 난수 생성 하한 인자
                                    radius_range_in_m[1])  # 반지름 난수 생성 상한 인자

    if mode == "spherical":  # 구면 랜덤 배치 모드 조건
        # 구면 위 랜덤 좌표 생성
        array_setup['mic_pos'] = place_random_points_on_sphere(n_mics, radius_in_m, min_dist_between_mics_in_m)
    elif mode == "poly2d":  # 평면 정다각형 배치 모드 조건
        # n_mics 꼭짓점 정다각형 생성 단계
        array_setup['mic_pos'] = generate_regular_polygon(n_mics, radius_in_m)

    return array_setup  # 생성 배열 설정 반환


# 구면 위 랜덤 점 배치 함수 정의
def place_random_points_on_sphere(n_points, radius_in_m, min_dist_between_points_in_m=0):
    points = np.zeros((n_points, 3))  # 점 좌표 저장 배열 초기화

    for i in range(n_points):  # 배치 대상 점 인덱스 반복, n_points개의 마이크 좌표 생성
        while True:  # 유효 후보 좌표 탐색 반복
            # 구면 좌표 각도 난수 생성 단계
            phi = np.random.uniform(0, 2 * np.pi)  # 방위각 난수
            theta = np.random.uniform(0, np.pi)  # 극각 난수

            # 구면 좌표의 직교 좌표 변환 단계
            x = np.cos(phi) * np.sin(theta)  # x축 단위 좌표
            y = np.sin(phi) * np.sin(theta)  # y축 단위 좌표
            z = np.cos(theta)  # z축 단위 좌표

            points[i] = radius_in_m*np.array([x, y, z])  # 반지름 적용 후보 좌표 저장

            if i == 0:  # 첫 번째 점 조건
                break  # 첫 번째 점 바로 확정

            # 후보 점과 기존 점 사이 거리 검증 단계
            dists = np.linalg.norm(points[:i] - points[i], axis=1)  # 기존 점과 후보 점 거리 계산
            if np.all(dists >= min_dist_between_points_in_m):  # 최소 거리 조건 충족 여부
                break  # 후보 점 확정 종료

    return points  # 점 좌표 배열 반환
