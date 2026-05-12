import numpy as np
import torch
import itertools


# 마이크 위치 특징 준비 함수
def prepare_mic_pos(mic_pos, mic_pair_idxs, mode="mic_positions"):
    batch_size, n_pairs = mic_pair_idxs.shape[:2]  # 배치 크기와 마이크 쌍 개수 추출

    out = torch.zeros(
        (batch_size, n_pairs, 2, 3), dtype=torch.float32, device=mic_pos.device  # 출력 텐서 크기와 속성 지정
    )  # 마이크 쌍별 좌표 저장 텐서 생성

    for i in range(batch_size):  # 배치 차원 반복
        for j in range(n_pairs):  # 마이크 쌍 차원 반복
            out[i, j, 0] = mic_pos[i, mic_pair_idxs[i, j, 0]]  # 첫 번째 마이크 좌표 저장
            out[i, j, 1] = mic_pos[i, mic_pair_idxs[i, j, 1]]  # 두 번째 마이크 좌표 저장

    if mode == "mic_positions":  # 원본 마이크 위치 특징 모드
        out = out.reshape(
            (batch_size, n_pairs, 6)  # 6개 값(마이크 2개 * xyz 좌표 3개)으로 펼치기 위한 형태
        ) * 100  # 미터 단위에서 센티미터 단위로 변환
    elif mode == "mic_diff_vector":  # 마이크 차이 벡터 모드
        out = out[:, :, 0] - out[:, :, 1]  # 두 마이크 좌표 차이 계산
    elif mode == "norm_mic_diff_vector":  # 정규화된 마이크 차이 벡터 모드
        # 두 마이크 사이의 차이 벡터 계산
        mic_diff = out[:, :, 0] - out[:, :, 1]  # 두 마이크 좌표 차이 계산
        # 차이 벡터의 크기 계산
        mic_diff_norm = torch.norm(mic_diff, dim=2, keepdim=True)  # 차이 벡터 norm 계산
        mic_diff = mic_diff / mic_diff_norm  # 차이 벡터 정규화
        # 정규화 벡터와 norm 연결
        out = torch.cat(
            (mic_diff, mic_diff_norm), dim=2  # 정규화 벡터와 노름 결합
        ).reshape((batch_size, n_pairs, 4))  # 4개 값(두 마이크 사이 정규화된 방향 벡터 3개 + 거리 1개)으로 펼치기 위한 형태

    return out  # 준비된 마이크 위치 특징 반환


# 모든 인덱스 쌍 생성 함수
def get_all_pairs(n, device=None):  
    "0부터 n-1까지의 모든 인덱스 쌍 반환"
    pairs = np.array(list(itertools.combinations(range(n), 2)))  # 가능한 모든 두 인덱스 조합 생성

    return torch.tensor(
        pairs,  # 텐서로 변환할 인덱스 쌍 배열
        dtype=torch.long, device=device  # 정수형 자료형과 대상 장치 지정
    )  # 인덱스 쌍 텐서 반환


# 공간 위치 인코딩 생성 함수
def create_spatial_positional_encoding(v, d, n=100):  
    """주어진 벡터 v의 공간 위치 인코딩을 생성.
    인코딩은 (n x d) 크기의 행렬이며, n은 인코딩의 주파수를
    제어하는 매개변수이고 d는 출력 인코딩의 차원.

    매개변수
    ----------
    v : torch.Tensor
        (batch_size, length) 형태의 인코딩할 벡터.
    d : int
        출력 인코딩의 차원.
    n : int = 100
        주파수 하이퍼파라미터.

    반환값
    -------
    torch.Tensor
        주어진 벡터의 공간 위치 인코딩.
    """

    batch_size, length = v.shape[:2]  # 배치 크기와 길이 추출

    # 위치 인코딩 계산
    pos_enc = torch.zeros(
        (batch_size, d), dtype=torch.float32, device=v.device  # 위치 인코딩 텐서 크기와 속성 지정
    )  # 위치 인코딩 저장 텐서 생성
    idxs = torch.arange(length, dtype=torch.float32, device=v.device)  # 위치 인덱스 텐서 생성

    for k in range(d):  # 출력 차원별 반복
        if k % 2 == 0:  # 짝수 차원 여부 확인
            func = torch.sin  # 사인 함수 선택
        else:  # 홀수 차원 분기
            func = torch.cos  # 코사인 함수 선택
        
        enc = func(
            idxs * v / (n ** (k / d))  # 위치와 입력 벡터를 이용한 주파수 스케일링
        )  # 차원별 위치 인코딩 계산
        pos_enc[:, k] = enc.mean(dim=1)  # 길이 차원 평균값 저장

    return pos_enc  # 공간 위치 인코딩 반환
