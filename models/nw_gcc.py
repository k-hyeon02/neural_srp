# 신경망 가중 일반화 상호 상관 (NW-GCC, Neurally Weighted Generalized Cross Correlation)

import torch.nn as nn

# 부모 클래스 GCC: 마이크 쌍 간의 일반화 상호 상관을 계산하는 모듈
from .signal_processing import GCC
# Mlp: 다층 퍼셉트론 — NW-GCC 가중치 예측에 사용
from .layers import Mlp


class NwGCC(GCC):
    """NW-GCC(신경망 가중 일반화 상호 상관)를 계산하는 모듈.

    기존 GCC에 신경망으로 학습된 0~1 사이의 가중치(likelihood)를 곱하여
    각 시간 지연 빈의 신뢰도를 동적으로 조절.
    가중치가 높을수록 해당 시간 지연 방향에 음원이 존재할 가능성이 높다고 판단.
    """

    def __init__(self, K, tau_max=None, transform=None, concat_bins=False, center=False):
        # 부모 클래스 GCC의 생성자 호출 — K(프레임 크기), tau_max(최대 시간 지연),
        # transform(PHAT 등 가중치 변환), concat_bins(시간 지연 인덱스 연결 여부),
        # center(영-중심 배열 여부)를 초기화
        super().__init__(K, tau_max, transform, concat_bins, center)

        # MLP 네트워크 정의: GCC 출력(2*tau_max 차원)을 입력받아 스칼라 가중치 1개를 출력
        self.net = Mlp(
            in_features=2 * self.tau_max,        # in_features  = 2 * tau_max: GCC 벡터의 길이 (양·음의 시간 지연 빈 수)
            out_features=1,                      # out_features = 1: 마이크 쌍별 단일 가중치 스칼라 출력
            hidden_features=2 * self.tau_max,    # hidden_features = 2 * tau_max: 은닉층 뉴런 수 (입력과 동일 크기)
            num_layers=2,                        # num_layers = 2: 입력층(Linear→ReLU) + 출력층(Linear→Sigmoid) 2개 블록
            activation=nn.ReLU(),                # activation = ReLU(): 은닉층 활성화 함수 — 음수 제거, 비선형성 부여
            batch_norm=False,                    # batch_norm = False: 배치 정규화 비활성화
            output_activation=nn.Sigmoid(),      # output_activation = Sigmoid(): 출력을 0~1 범위로 압축 — 확률적 가중치로 해석
        ) # TODO: make this configurable

    def forward(self, x, tau_max=None):

        # 부모 클래스 GCC.forward로 마이크 쌍별 GCC 계산
        # 반환 shape: [batch_size, n_frames, n_mics, n_mics, 2*tau_max]
        gcc = super().forward(x)

        # (비활성) GCC를 최댓값으로 나눠 [-1, 1] 범위로 정규화하는 코드 — 현재 미사용
        # gcc /= gcc.abs().max(dim=-1, keepdim=True)[0]

        # gcc의 shape 분해: B=배치, F=프레임, M=마이크 수, _=M(쌍 축), _=2*tau_max(시간 지연 빈)
        batch_size, n_frames, n_mics, _, _ = gcc.shape

        # MLP로 GCC 벡터(2*tau_max)에서 마이크 쌍별 가중치(스칼라) 예측
        # 입력 shape:  [B, F, M, M, 2*tau_max]
        # 출력 shape:  [B, F, M, M, 1]  — 마지막 차원이 1인 스칼라 가중치
        weight = self.net(gcc)

        # GCC에 가중치를 원소별 곱(broadcasting)으로 적용하여 반환
        # weight의 마지막 차원(1)이 gcc의 2*tau_max 차원에 자동 브로드캐스트됨
        # 출력 shape: [B, F, M, M, 2*tau_max] — 가중치가 적용된 NW-GCC
        return gcc * weight
