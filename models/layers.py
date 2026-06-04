"""
	DOA(음원 방향 추정)를 위한 PyTorch 함수 및 레이어 모음.

	파일명: acousticTrackingModules.py
	작성자: David Diaz-Guerra
	작성일: 05/2020
	Python 버전: 3.8
	Pytorch 버전: 1.4.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausConv3d(nn.Module):
    """SRP-PHAT 맵 시퀀스를 위한 인과적(Causal) 3D 합성곱 레이어.
    미래 정보를 참조하지 않도록 시간 축 방향으로만 패딩을 적용."""

    def __init__(self, in_channels, out_channels, kernel_size):
        # nn.Module 부모 클래스 초기화
        super().__init__()
        # 시간 축 인과성 보장을 위한 패딩 크기: 커널 시간 크기 - 1
        # 현재 시점에서 미래 프레임을 참조하지 않도록 강제 - 인과적 합성곱
        self.pad = kernel_size[0] - 1
        # 3D 합성곱 레이어 정의: 시간 축에만 self.pad만큼 패딩, 나머지 축은 패딩 없음
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, padding=(self.pad, 0, 0)
        )

    def forward(self, x):
        # 합성곱 후 시간 축의 뒤쪽 self.pad 개를 제거하여 미래 정보 누설 방지
        # 입력 shape: (B, C_in, T, H, W) → 출력 shape: (B, C_out, T, H', W')
        return self.conv(x)[:, :, : -self.pad, :, :]


class CausConv2d(nn.Module):
    """스펙트로그램 및 GCC 시퀀스를 위한 인과적(Causal) 2D 합성곱 레이어.
    미래 정보를 참조하지 않도록 시간 축 방향으로만 패딩을 적용."""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        # 시간 축 인과성 보장을 위한 패딩 크기: 커널 시간 크기 - 1
        self.pad = kernel_size[0] - 1
        # 2D 합성곱 레이어 정의: 시간 축에만 self.pad만큼 패딩, 주파수 축은 패딩 없음
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=(self.pad, 0)
        )

    def forward(self, x):
        # 합성곱 후 시간 축의 뒤쪽 self.pad 개를 제거하여 미래 정보 누설 방지
        # 입력 shape: (B, C_in, T, F) → 출력 shape: (B, C_out, T, F')
        return self.conv(x)[:, :, : -self.pad, :]


class CausConv1d(nn.Module):
    """인과적(Causal) 1D 합성곱 레이어.
    미래 정보를 참조하지 않도록 시간 축 방향으로만 패딩을 적용.
    팽창(dilation) 합성곱도 지원."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        # 팽창 합성곱을 고려한 인과성 패딩 크기: (커널 크기 - 1) * dilation
        self.pad = (kernel_size - 1) * dilation
        # 1D 합성곱 레이어 정의: self.pad만큼 패딩, dilation 적용
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=self.pad, dilation=dilation
        )

    def forward(self, x):
        # 합성곱 후 시간 축의 뒤쪽 self.pad 개를 제거하여 미래 정보 누설 방지
        # 입력 shape: (B, C_in, T) → 출력 shape: (B, C_out, T)
        return self.conv(x)[:, :, : -self.pad]


class SphericPad(nn.Module):
    """구면 좌표계에 적합한 패딩 모듈.
    시간 축: 복제(replication) 패딩, 고도(elevation) 축: 반사(reflect) 패딩,
    방위각(azimuth) 축: 원형(circular) 패딩을 각각 적용한다.
    시간 축 패딩은 선택 사항이며, CausConv3d와 함께 사용하지 말 것.
    """

    def __init__(self, pad):
        super().__init__()

        if len(pad) == 4:
            # 패딩값이 4개인 경우: 방위각/고도 축 패딩만 설정 (공간 패딩 전용)
            self.padLeft, self.padRight, self.padTop, self.padBottom = pad
            # 시간 축 패딩은 사용하지 않으므로 0으로 초기화
            self.padFront, self.padBack = 0, 0
        elif len(pad) == 6:
            # 패딩값이 6개인 경우: 방위각/고도/시간 축 패딩 모두 설정
            (
                self.padLeft,    # 방위각 좌측 패딩 크기
                self.padRight,   # 방위각 우측 패딩 크기
                self.padTop,     # 고도 상단 패딩 크기
                self.padBottom,  # 고도 하단 패딩 크기
                self.padFront,   # 시간 축 앞쪽 패딩 크기
                self.padBack,    # 시간 축 뒤쪽 패딩 크기
            ) = pad
        else:
            # 4개 또는 6개가 아닌 경우 예외 발생
            raise Exception(
                "Expect 4 or 6 values for padding (padLeft, padRight, padTop, padBottom, [padFront, padBack])"
            )

    def forward(self, x):
        # 방위각 축 패딩 크기가 입력 크기를 초과하지 않는지 검증
        assert (
            x.shape[-1] >= self.padRight and x.shape[-1] >= self.padLeft
        ), "Padding size should be less than the corresponding input dimension for the azimuth axis"

        # 시간 축 패딩
        if self.padBack > 0 or self.padFront > 0:
            # 시간 축 패딩이 필요한 경우: 복제(replication) 패딩 적용
            # F.pad의 패딩 순서는 마지막 차원부터 역순: (맨끝, 맨끝, ..., 시간앞, 시간뒤)
            x = F.pad(x, (0, 0, 0, 0, self.padFront, self.padBack), "replicate")

        # 이후 연산에서 원래 shape 복원을 위해 저장
        input_shape = x.shape
        # 배치 차원을 유지하고 중간 차원들을 하나로 병합: (B, C, T, H, W) → (B, C*T, H, W)
        # B: 배치(한 번에 처리하는 샘플) 수, C: 마이크 쌍 수, T: 프레임 수, H: elevation 격자 수, W: azimuth 격자 수
        # F.pad가 4D 텐서까지만 지원하기 때문
        x = x.view((x.shape[0], -1, x.shape[-2], x.shape[-1]))

        # 고도(elevation) 축에 반사(reflect) 패딩 적용
        # 수학적으로는 pi(180도) 위상 이동이 필요하나 현재는 단순 반사만 적용
        x = F.pad(
            x, (0, 0, self.padTop, self.padBottom), "reflect"
        )  # 실제로는 pi 위상 이동이 추가되어야 함

        # 방위각(azimuth) 축에 원형(circular) 패딩 적용: 끝 부분을 앞에, 앞 부분을 끝에 이어 붙임
        # shape: (..., W) → (..., padLeft + W + padRight)
        x = torch.cat((x[..., -self.padLeft :], x, x[..., : self.padRight]), dim=-1)

        # 병합했던 중간 차원들을 원래 shape으로 복원하여 반환
        # shape: (B, C*T, H_padded, W_padded) → (B, C, T, H_padded, W_padded)
        return x.view((x.shape[0],) + input_shape[1:-2] + (x.shape[-2], x.shape[-1]))


class Mlp(nn.Module):
    """다층 퍼셉트론(Multi-layer Perceptron) 모듈.
    레이어 수, 활성화 함수, 드롭아웃, 배치 정규화를 유연하게 설정 가능."""

    def __init__(self, in_features, out_features, hidden_features,
                 num_layers, activation="relu", dropout=0, batch_norm=False,
                 output_activation=None, dtype=torch.float32):
        # nn.Module 부모 클래스 초기화
        super().__init__()

        # 각 MLP 블록(Linear + BN + Activation + Dropout)을 담는 ModuleList
        self.blocks = nn.ModuleList()
        # 레이어 파라미터의 데이터 타입 저장 (기본값: float32)
        self.dtype = dtype

        if num_layers == 1:
            # 단일 레이어인 경우: in_features → out_features 직접 연결
            self.blocks.append(
                self._create_mlp_block(in_features, out_features,
                                       batch_norm, activation, dropout)
            )
        else:
            # 다중 레이어인 경우: num_layers개의 블록을 순서대로 생성
            for i in range(num_layers):
                if i == 0: # 첫 번째 레이어: 입력 차원 → 은닉 차원
                    in_features_i = in_features       # 입력 피처 수
                    out_features_i = hidden_features  # 은닉 피처 수
                    activation = activation           # 중간 레이어 활성화 함수 유지
                    
                elif i < num_layers - 1: # 중간 은닉 레이어: 은닉 차원 → 은닉 차원
                    in_features_i = hidden_features   # 은닉 피처 수
                    out_features_i = hidden_features  # 은닉 피처 수
                    activation = activation           # 중간 레이어 활성화 함수 유지

                else: # 마지막 레이어: 은닉 차원 → 출력 차원
                    in_features_i = hidden_features   # 은닉 피처 수
                    out_features_i = out_features     # 최종 출력 피처 수
                    activation = output_activation    # 출력 레이어 전용 활성화 함수 적용

                # i번째 MLP 블록 생성 후 blocks에 추가
                self.blocks.append(
                    self._create_mlp_block(in_features_i, out_features_i,
                                            batch_norm, activation, dropout)
                )

    def _create_mlp_block(self, in_features, out_features, batch_norm, activation, dropout):
        # 단일 블록을 구성하는 레이어들을 담는 ModuleList
        layers = nn.ModuleList()
        # 선형 변환 레이어 추가: y = xW^T + b, shape: (..., in_features) → (..., out_features)
        layers.append(nn.Linear(in_features, out_features, dtype=self.dtype))

        if batch_norm:
            # 배치 정규화 레이어 추가: 학습 안정화 및 과적합 억제
            layers.append(nn.BatchNorm1d(out_features, dtype=self.dtype))

        if activation is not None:
            # 활성화 함수가 지정된 경우에만 추가
            if activation == "relu":
                # 문자열 "relu" 입력 시 ReLU 활성화 함수로 변환: max(0, x)
                activation = nn.ReLU()
            elif activation == "prelu":
                # 문자열 "prelu" 입력 시 PReLU 활성화 함수로 변환: max(ax, x), a는 학습 파라미터
                activation = nn.PReLU()
            elif isinstance(activation, nn.Module):
                # 이미 nn.Module 인스턴스인 경우 그대로 사용
                pass
            # 결정된 활성화 함수 레이어 추가
            layers.append(activation)

        if dropout > 0:
            # 드롭아웃 비율이 0보다 큰 경우에만 Dropout 레이어 추가 (과적합 방지)
            layers.append(nn.Dropout(dropout))
        # ModuleList를 Sequential로 변환하여 반환 (순서대로 실행 가능)
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 모든 블록을 순서대로 순전파 수행
        for block in self.blocks:
            # 각 블록 내의 레이어를 순서대로 적용
            for layer in block:
                if isinstance(layer, nn.BatchNorm1d):
                    # BatchNorm1d는 피처 차원이 두 번째 차원(dim=1)에 있어야 하는 반면,
                    # Linear 레이어는 피처 차원이 마지막 차원에 있으므로 차원 순서를 조정해야 함
                    if len(x.shape) == 3:
                        # 3D 입력 (B, T, F): 전치(transpose)로 (B, F, T)로 변환 후 BN 적용, 다시 원복
                        # (B, T, F) → transpose → (B, F, T) → BN → (B, F, T) → transpose → (B, T, F)
                        x = layer(x.transpose(1, 2)).transpose(1, 2)
                    elif len(x.shape) == 2:
                        # 2D 입력 (B, F): BN이 바로 dim=1(피처 차원)에 적용되므로 변환 불필요
                        x = layer(x)
                    else:
                        # 2D 또는 3D가 아닌 경우 예외 발생
                        raise Exception("Input should be 2D or 3D")
                else:
                    # BatchNorm1d가 아닌 레이어(Linear, ReLU, Dropout 등)는 그대로 적용
                    x = layer(x)

        # 최종 출력 반환: shape은 마지막 블록의 out_features에 따라 결정됨
        return x
