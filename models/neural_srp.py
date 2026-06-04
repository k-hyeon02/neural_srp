import numpy as np
import torch
import torch.nn as nn

from datasets.mic_pos_utils import get_all_pairs, prepare_mic_pos
from models.signal_processing import GCC, Window
from models.layers import Mlp
from models.mic_selection import select_pairs


class ConvBlock(nn.Module):
    """기본 Conv2d 블록: Conv → BN → PReLU → (MaxPool) → (Dropout)"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        pool_size=None,
        dropout_rate=0.0,
    ):

        super().__init__()

        # 2D 합성곱 레이어: 공간/주파수 피처 추출
        # shape: (B, in_channels, T, F) → (B, out_channels, T', F')
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # 배치 정규화: 채널 방향(out_channels)으로 정규화하여 학습 안정화
        self.bn = nn.BatchNorm2d(out_channels)
        # PReLU: max(a*x, x), a는 학습 파라미터 — ReLU의 음수 기울기 소실 문제 보완
        self.activation = nn.PReLU()

        if pool_size is not None:
            # 풀링 크기가 주어진 경우에만 MaxPool2d 추가 (시간·주파수 축 다운샘플링)
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

        if dropout_rate > 0.0:
            # 드롭아웃 비율이 양수인 경우에만 Dropout2d 추가 (채널 단위 드롭아웃, 과적합 방지)
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        # Conv → BN → PReLU 순으로 순전파
        # shape: (B, in_channels, T, F) → (B, out_channels, T, F)
        x = self.activation(self.bn(self.conv(x)))

        if hasattr(self, "pool"):
            # MaxPool로 공간 차원(T, F) 축소: 계산량 감소 + 위치 불변성 획득
            x = self.pool(x)

        if hasattr(self, "dropout"):
            # Dropout2d: 채널 전체를 랜덤하게 0으로 설정
            x = self.dropout(x)

        return x


class ConditionalConvBlock(nn.Module):
    """마이크 메타데이터를 바이어스로 주입하는 조건부 Conv2d 블록.

    메타데이터(마이크 위치 등)를 Linear로 투영한 뒤 Conv 출력에 더해
    'early_conv_bias' 방식으로 공간 정보를 피처 추출 단계에서 결합.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_metadata=0,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        pool_size=None,
        dropout_rate=0.0,
    ):

        super().__init__()

        # 기본 2D 합성곱 레이어
        # shape: (B, in_channels, T, F) → (B, out_channels, T', F')
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # 배치 정규화 (out_channels 차원)
        self.bn = nn.BatchNorm2d(out_channels)
        # PReLU 활성화 함수
        self.activation = nn.PReLU()

        if pool_size is not None:
            # 시간·주파수 축 다운샘플링용 MaxPool2d
            self.pool = nn.MaxPool2d(kernel_size=pool_size)

        if dropout_rate > 0.0:
            # 채널 단위 Dropout2d
            self.dropout = nn.Dropout2d(p=dropout_rate)

        if n_metadata > 0:
            # 메타데이터(마이크 위치 벡터 등)를 채널 수만큼 투영하는 Linear 레이어
            # n_metadata → out_channels: 각 채널에 메타데이터 바이어스로 더할 수 있도록 차원 맞춤
            self.metadata_proj = nn.Linear(n_metadata, out_channels)

    def forward(self, x, metadata=None):

        # Conv 연산 수행 (BN·활성화 전에 분리한 이유: 메타데이터 바이어스를 중간에 삽입)
        # shape: (B, in_channels, T, F) → (B, out_channels, T, F)
        x = self.conv(x)

        # 메타데이터를 채널 바이어스로 더하는 early fusion
        if metadata is not None and hasattr(self, "metadata_proj"):
            # metadata: (B, n_metadata) → Linear → (B, out_channels)
            metadata = self.metadata_proj(metadata)
            # 공간 차원(T, F)에 브로드캐스트하기 위해 축 2개 추가
            # shape: (B, out_channels) → (B, out_channels, 1, 1)
            metadata = metadata.unsqueeze(-1).unsqueeze(-1)
            # Conv 출력에 메타데이터 바이어스를 더해 마이크 위치 정보 주입
            # 브로드캐스트: (B, out_channels, 1, 1) → (B, out_channels, T, F)
            x = x + metadata

        # BN → PReLU 적용 (메타데이터 바이어스 포함 후 정규화)
        x = self.activation(self.bn(x))

        if hasattr(self, "pool"):
            x = self.pool(x)

        if hasattr(self, "dropout"):
            x = self.dropout(x)

        return x


class NeuralSrp(nn.Module):
    def __init__(self, n_gcc_bins, params, n_max_sources=2, n_max_dataset_sources=2):
        """
        CRNN 모델을 초기화.
        :param n_gcc_bins: 입력 GCC 빈(bin)의 수.
        :param params: 하이퍼파라미터 딕셔너리.
        :param n_max_sources: 동시에 추적 가능한 최대 음원 수.
        """

        super().__init__()

        # 모델이 동시에 추적할 수 있는 최대 음원 수 (출력 DOA 벡터 수 결정)
        self.n_max_sources = n_max_sources

        # 실제 데이터셋에서 사용할 음원 수 (n_max_sources보다 작거나 같을 수 있음)
        self.n_max_dataset_sources = n_max_dataset_sources

        # True면 DOA와 함께 음원 활성화(activity) 출력도 생성
        self.use_activity_out = params["use_activity_output"]

        # True면 GRU를 양방향(BiGRU)으로 사용 (전후 맥락 모두 활용)
        self.bidirectional_rnn = params["bidirectional_rnn"]

        # 마이크 쌍 피처 집계 방식: "sum", "mean", "prod" 중 하나
        self.pair_agg_mode = params["pair_agg_mode"]

        # True면 은닉 레이어에 배치 정규화 적용
        self.use_batch_norm_hidden = params["use_batch_norm_hidden"]

        # 여러 ConvBlock을 담는 ModuleList (가변 개수의 합성곱 레이어 지원)
        self.conv_block_list = nn.ModuleList()

        # 마이크 메타데이터 종류: "mic_positions", "mic_diff_vector",
        # "norm_mic_diff_vector", "idx" 중 하나
        self.metadata_type = params["metadata_type"]

        # 메타데이터 융합 방식: "early_conv_bias"(Conv 내부 바이어스) 또는
        # "late_concat"(GRU 이후 연결)
        self.metadata_fusion_mode = params["metadata_fusion_mode"]
        assert self.metadata_type in [
            "mic_positions",
            "mic_diff_vector",
            "norm_mic_diff_vector",
            "idx",
        ]
        assert self.metadata_fusion_mode in [
            "early_conv_bias",
            "late_concat",
        ]

        # Conv 출력 집계 방식: "flatten"(펼치기), "sum", "mean", "prod", "max"
        self.conv_agg_mode = params["conv_agg_mode"]

        if params["input_feature"] == "gcc":
            # GCC 피처: 상호상관 값 1채널 입력
            self.n_input_channels = 1  # Correlation values
            # Also tried 2 (respective bins)
        elif params["input_feature"] == "pairwise_mel_phase":
            # 쌍별 Mel 위상 피처: 마이크당 1채널씩 총 2채널 입력
            self.n_input_channels = 2  # 1 for each mic

        # 메타데이터 벡터의 차원 수 (metadata_type에 따라 결정)
        if self.metadata_type == "idx":
            self.n_metadata = 1  # 마이크 쌍 인덱스 (스칼라)
        elif self.metadata_type == "mic_positions":
            self.n_metadata = 6  # 각 마이크 3D 좌표 2개 = 6
        elif self.metadata_type == "norm_mic_diff_vector":
            self.n_metadata = 4  # 단위 차이 벡터(3) + 거리(1) = 4
        elif self.metadata_type == "mic_diff_vector":
            self.n_metadata = 3  # 마이크1 - 마이크2 의 3D 벡터

        # ── 입력 배치 정규화 ──
        self.use_batch_norm_input = params["use_batch_norm_input"]
        if self.use_batch_norm_input:
            # 신호 피처(채널 방향) 정규화: (B, n_input_channels, T, F) 형태에 적용
            self.input_bn = nn.BatchNorm2d(self.n_input_channels)
            # 메타데이터 정규화: (B, 1, n_metadata) → unsqueeze 후 적용
            self.metadata_bn = nn.BatchNorm1d(1)

        # ── 합성곱 블록 스택 구성 ──
        n_conv_input = self.n_input_channels

        # f_pool_size 리스트 길이 = 합성곱 레이어 수
        n_conv_layers = len(params["f_pool_size"])
        for conv_cnt in range(n_conv_layers):
            self.conv_block_list.append(
                ConditionalConvBlock(
                    # 첫 번째 레이어는 원본 채널 수, 이후는 nb_cnn2d_filt 채널 유지
                    in_channels=params["nb_cnn2d_filt"] if conv_cnt else n_conv_input,
                    out_channels=params["nb_cnn2d_filt"],
                    # early_conv_bias 모드일 때만 ConditionalConvBlock에 메타데이터 크기 전달
                    n_metadata=(
                        self.n_metadata
                        if self.metadata_fusion_mode == "early_conv_bias"
                        else 0
                    ),
                    # (t_pool_size, f_pool_size): 시간·주파수 축 풀링 비율
                    pool_size=(
                        params["t_pool_size"][conv_cnt],
                        params["f_pool_size"][conv_cnt],
                    ),
                    dropout_rate=params["dropout_rate"],
                )
            )

        # GRU 은닉 유닛 수
        self.n_rnn_features = params["rnn_size"]

        if self.conv_agg_mode == "flatten":
            # Conv 출력을 펼칠 때의 GRU 입력 크기
            # = 필터 수 × (주파수 빈 수 / 주파수 풀링 비율의 곱)
            self.in_gru_size = int(
                params["nb_cnn2d_filt"] * (n_gcc_bins / np.prod(params["f_pool_size"]))
            )
        elif self.conv_agg_mode in ["sum", "mean", "prod", "max"]:
            # Conv 출력을 집계할 때의 GRU 입력 크기 = 필터 수만
            self.in_gru_size = params["nb_cnn2d_filt"]

        # GRU 레이어: 시계열(프레임) 방향 순환 처리
        # input_size = in_gru_size, hidden_size = n_rnn_features
        # batch_first=True: (B, T, F) 형태 입력 허용
        self.gru = nn.GRU(
            input_size=self.in_gru_size,
            hidden_size=self.n_rnn_features,
            num_layers=params["nb_rnn_layers"],
            batch_first=True,
            dropout=params["dropout_rate"],
            bidirectional=params["bidirectional_rnn"],
        )

        # DOA 출력 크기: 음원당 (x, y, z) 3좌표 × 최대 음원 수
        self.n_output = n_max_sources * 3  # 3 coordinates (x, y, z) for each source

        # 쌍별 FNN의 입력 피처 크기 (기본: GRU 은닉 크기)
        self.n_pairwise_input_features = self.n_rnn_features

        # 쌍별 FNN의 출력 피처 크기
        self.n_pairwise_output_features = params["fnn_pairwise_size"]
        if self.metadata_fusion_mode == "late_concat":
            # late_concat 모드: GRU 출력 뒤에 메타데이터를 연결하므로 입력 크기 증가
            self.n_pairwise_input_features += self.n_metadata

        if params["nb_pairwise_fnn_layers"] > 0:
            # 쌍별 MLP: 각 마이크 쌍의 GRU 출력을 압축·변환
            self.fnn_pairwise = Mlp(
                self.n_pairwise_input_features,
                self.n_pairwise_output_features,
                self.n_pairwise_output_features,
                params["nb_pairwise_fnn_layers"],
                activation="prelu",
                output_activation="prelu",
                batch_norm=params["use_batch_norm_hidden"],
            )
        else:
            # 레이어 수가 0이면 항등 함수로 대체 (pass-through)
            self.fnn_pairwise = nn.Identity()
            # Identity 사용 시 출력 크기 = 입력 크기
            self.n_pairwise_output_features = self.n_pairwise_input_features

        if self.use_batch_norm_hidden:
            # 쌍별 FNN 출력에 대한 배치 정규화
            # BatchNorm1d는 피처 차원이 dim=1이어야 하므로 forward에서 transpose 필요
            # (B×15, T, 128) -> (B×15, 128, T) -> BN 후 다시 (B×15, T, 128)로 복원
            self.pairwise_bn = nn.BatchNorm1d(self.n_pairwise_output_features)

        # DOA 추정 MLP: 집계된 쌍별 피처 → (x, y, z) × n_max_sources
        # 출력 활성화: Tanh → [-1, 1] 범위의 단위구면 좌표 출력
        self.fnn_doa = Mlp(
            self.n_pairwise_output_features,
            self.n_output,
            params["fnn_doa_size"],
            params["nb_fnn_layers"],
            activation="prelu",
            output_activation=nn.Tanh(),
            batch_norm=params["use_batch_norm_hidden"],
        )

        # ── 음원 활성화(activity) 감지 브랜치 ──
        if self.use_activity_out:
            # 활성화 MLP: 집계된 쌍별 피처 → 각 음원의 존재 여부(로짓)
            # output_activation=None: BCEWithLogitsLoss 등을 외부에서 적용
            self.fnn_activity = Mlp(
                self.n_pairwise_output_features,
                n_max_sources,
                params["fnn_act_size"],
                params["nb_fnn_act_layers"],
                activation="prelu",
                output_activation=None,
                batch_norm=params["use_batch_norm_hidden"],
            )

        # summary(self)

    def _pairwise_forward(self, x, metadata=None):
        """마이크 쌍 하나에 대한 CRNN 순전파.
        마이크 쌍의 GCC feature를 입력 받아 압축된 feature vector로 변환

        Args:
            x:        (batch_size, time_steps, n_features, n_channels)
            metadata: (batch_size, n_metadata)
        Returns:
            x:        (batch_size, time_steps, n_pairwise_output_features)
        """

        # 채널 축을 두 번째(dim=1)로 이동: Conv2d 입력 형식 맞춤
        # (B, T, F, C) → (B, C, T, F)
        x = x.moveaxis(3, 1)
        if self.use_batch_norm_input:
            # 입력 신호 배치 정규화: (B, C, T, F) 형태에서 C 차원으로 정규화
            x = self.input_bn(x)
            # 메타데이터 배치 정규화:
            # (B, n_metadata) → unsqueeze(1) → (B, 1, n_metadata) → BN → [:, 0] → (B, n_metadata)
            metadata = self.metadata_bn(metadata.unsqueeze(1))[:, 0]

        # ── 1. 합성곱 블록 스택 순전파 ──
        for conv_block in self.conv_block_list:
            # 각 ConditionalConvBlock에 신호와 메타데이터를 함께 전달
            # early_conv_bias 모드: 블록 내부에서 메타데이터를 바이어스로 더함
            x = conv_block(x, metadata=metadata)
        # 이후 shape: (B, nb_cnn2d_filt, T', F')

        # 시간 축과 피처 맵 축 교환: Conv 출력 → GRU 입력 형식
        # (B, nb_cnn2d_filt, T', F') → (B, T', nb_cnn2d_filt, F')
        x = x.transpose(1, 2).contiguous()

        if self.conv_agg_mode == "flatten":
            # 주파수 축과 피처 맵 축을 하나로 합침
            # (B, T', nb_cnn2d_filt, F') → (B, T', nb_cnn2d_filt * F')
            x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        elif self.conv_agg_mode == "sum":
            # 주파수 축에 대해 합산: (B, T', nb_cnn2d_filt, F') → (B, T', nb_cnn2d_filt)
            x = x.sum(dim=-1)
        elif self.conv_agg_mode == "mean":
            # 주파수 축에 대해 평균: (B, T', nb_cnn2d_filt, F') → (B, T', nb_cnn2d_filt)
            x = x.mean(dim=-1)
        elif self.conv_agg_mode == "prod":
            # 주파수 축에 대해 곱: (B, T', nb_cnn2d_filt, F') → (B, T', nb_cnn2d_filt)
            x = x.prod(dim=-1)
        elif self.conv_agg_mode == "max":
            # 주파수 축에 대해 최댓값: (B, T', nb_cnn2d_filt, F') → (B, T', nb_cnn2d_filt)
            x = x.max(dim=-1)[0]

        # 이후 shape: (B, T', in_gru_size)

        # ── 2. GRU 순전파 ──
        # GRU 출력: (B, T', n_rnn_features) 또는 양방향이면 (B, T', 2*n_rnn_features)
        x, _ = self.gru(x)
        # Tanh 적용으로 출력을 [-1, 1] 범위로 정규화
        x = torch.tanh(x)

        if self.bidirectional_rnn:
            # 양방향 GRU 출력을 원소별 곱으로 결합 (forward × backward)
            # 앞 절반: forward, 뒷 절반: backward
            # shape: (B, T', 2*n_rnn_features) → (B, T', n_rnn_features)
            x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]
            # concat 대신 곱을 쓰는 이유:
            # forward와 backward 둘 다 "강하게 활성화"된 피처만 살아남음
            # → 둘 다 동의하는 패턴만 강조하는 효과
            # → 차원을 늘리지 않아서 이후 레이어 크기도 그대로 유지

        # ── 3. 메타데이터 late_concat(선택적): 마이크 위치 정보 결합 ──
        if metadata is not None and self.metadata_fusion_mode == "late_concat":
            # 메타데이터를 시간 축에 걸쳐 반복하여 GRU 출력과 연결
            # metadata.unsqueeze(1): (B, n_metadata) → (B, 1, n_metadata), 시간 차원 추가
            # (B×15, T/5, 6) 짜리 전부 1인 텐서 * metadata(B*15, 1, 6): broadcast로 (B*15, T/5, 6) 모든 프레임에 동일한 matadata 값
            metadata = torch.ones((x.shape[0], x.shape[1], self.n_metadata)).to(x.device) * metadata.unsqueeze(1)
            # GRU 출력과 메타데이터를 피처 차원으로 연결
            # (B, T', n_rnn_features) + (B, T', n_metadata) → (B, T', n_rnn_features + n_metadata)
            x = torch.cat((x, metadata), dim=2)

        # ── 4. 쌍별 완전연결 레이어 ──
        # (B, T', n_pairwise_input_features) → (B, T', n_pairwise_output_features)
        x = self.fnn_pairwise(x)
        if self.use_batch_norm_hidden:
            # BatchNorm1d는 (B, F, T) 형식이므로 transpose로 피처 축을 dim=1로 이동
            # (B, T', F) → transpose → (B, F, T') → BN → (B, F, T') → transpose → (B, T', F)
            x = self.pairwise_bn(x.transpose(1, 2)).transpose(1, 2)

        return x  # shape: (B, T', n_pairwise_output_features)

    def forward(self, x):
        """네트워크의 순전파.

        Args:
            x (dict): "signal"과 "mic_pos" 키를 포함하는 딕셔너리.
                "signal": 입력 신호,
                shape = (batch_size, n_mic_pairs, n_samples, n_gcc_bins, n_channels).
                "mic_pos": 마이크 위치 메타데이터,
                shape = (batch_size, n_mic_pairs, n_metadata).
        Returns:
            torch.Tensor: 네트워크 출력값
        """
        # 딕셔너리에서 메타데이터와 신호를 분리
        metadata = x["mic_pos"]  # shape: (B, n_mic_pairs, n_metadata)
        x = x["signal"]  # shape: (B, n_mic_pairs, n_frames, n_gcc_bins, n_channels)

        batch_size, n_mic_pairs, n_frames, n_features, n_channels = x.shape

        # ── 1. 쌍별 CRNN 적용 ──
        # 배치 차원과 마이크 쌍 차원을 합쳐 병렬 처리 (같은 CRNN 가중치 공유)
        # (B, n_mic_pairs, T, F, C) → (B*n_mic_pairs, T, F, C)
        x = x.reshape(-1, n_frames, n_features, n_channels)
        # (B, n_mic_pairs, n_metadata) → (B*n_mic_pairs, n_metadata)
        metadata = metadata.reshape(-1, self.n_metadata)

        # 쌍별 CRNN 순전파: 모든 쌍에 동일한 가중치 공유
        x = self._pairwise_forward(x, metadata=metadata)
        # 배치와 쌍 차원 복원
        # (B*n_mic_pairs, T', n_pairwise_output_features) → (B, n_mic_pairs, T', n_pairwise_output_features)
        x = x.reshape(batch_size, n_mic_pairs, -1, self.n_pairwise_output_features)

        # 마이크 쌍 차원(dim=1)에 대해 집계: 모든 쌍의 피처를 하나로 합산
        if self.pair_agg_mode == "sum":
            # (B, n_mic_pairs, T', F) → (B, T', F)
            x = x.sum(dim=1)
        if self.pair_agg_mode == "mean":
            # (B, n_mic_pairs, T', F) → (B, T', F)
            x = x.mean(dim=1)
        elif self.pair_agg_mode == "prod":
            # (B, n_mic_pairs, T', F) → (B, T', F)
            x = x.prod(dim=1)

        # activity 브랜치와 DOA 브랜치가 같은 입력을 공유하므로 저장
        x_0 = x  # shape: (B, T', n_pairwise_output_features)

        # ── 2. DOA 추정 브랜치 ──
        # (B, T', n_pairwise_output_features) → (B, T', n_max_sources*3)
        doa = self.fnn_doa(x)

        # 출력을 (B, T', n_max_sources, 3) 형태로 변환
        # view: (B, T', 3, n_max_sources) → transpose(-1, -2): (B, T', n_max_sources, 3)
        doa = doa.view(doa.shape[0], doa.shape[1], 3, self.n_max_sources).transpose(
            -1, -2
        )

        # 실제 데이터셋 음원 수만큼 슬라이싱 (n_max_sources >= n_max_dataset_sources)
        doa = doa[:, :, : self.n_max_dataset_sources]

        if doa.shape[2] == 1:
            # 음원이 1개뿐이면 음원 차원 제거: (B, T', 1, 3) → (B, T', 3)
            doa = doa[:, :, 0]

        out = {"doa_cart": doa}  # 단위구면 데카르트 좌표 (Tanh 출력, 범위 [-1, 1])

        # ── 3. 음원 활성화 감지 브랜치 (선택적) ──
        if self.use_activity_out:
            # (B, T', n_pairwise_output_features) → (B, T', n_max_sources)
            activity = self.fnn_activity(x_0)

            # 실제 데이터셋 음원 수만큼 슬라이싱
            activity = activity[:, :, : self.n_max_dataset_sources]
            # if activity.shape[2] == 1:
            #     # Squeeze the source dimension if there is only one source
            #     activity = activity[:, :, 0]

            out["activity"] = activity  # 음원 존재 여부 로짓

        return out


class NeuralSrpFeatureExtractor(torch.nn.Module):
    """오디오 신호 → NeuralSrp 입력 피처 변환 파이프라인.

    1. Window: 신호를 겹치는 프레임으로 분할 + 윈도우 함수 적용
    2. GCC: 마이크 쌍별 일반화 상호상관(GCC-PHAT) 계산
    3. _sample_pairs: 학습/추론에 사용할 마이크 쌍 서브샘플링
    """

    def __init__(self, params):
        super().__init__()

        # 마이크 쌍 샘플링 방식: "all", "random", "first", "distinct_angles"
        self.mic_pair_sampling_mode = params["mic_pair_sampling_mode"]
        # 한 번에 사용할 마이크 쌍 수
        self.n_mic_pairs = params["n_mic_pairs"]
        # NeuralSrp에 전달할 메타데이터 종류
        self.metadata_type = params["neural_srp"]["metadata_type"]

        # ── 1. 윈도우 변환 모듈 ──
        # 신호를 win_size 샘플 크기, hop_size 간격으로 프레임 분할 + Hann 윈도우 적용
        self.window = Window(
            params["win_size"],
            int(
                params["win_size"] * params["hop_rate"]
            ),  # hop_size = win_size * hop_rate
            window="hann",
        )

        # ── 2. GCC-PHAT 피처 추출기 ──
        # tau_max = nb_gcc_bins // 2: 출력 GCC 빈 수 = 2 * tau_max = nb_gcc_bins
        # concat_bins=False: 시간 지연 인덱스 없이 GCC 값만 출력
        # center=True: GCC를 [-tau_max, ..., tau_max-1] 순서로 중앙 정렬
        self.feature_extractor = GCC(
            params["win_size"],
            tau_max=params["nb_gcc_bins"] // 2,
            transform="phat",
            concat_bins=False,
            center=True,
        )

    def forward(self, x):
        """
        x = {
          "signal":  (B, T_samples, M)     ← 원본 오디오 파형
          "mic_pos": (B, M, 3)             ← 마이크 3D 좌표
        }
        """
        # 신호 프레임 분할 + Hann 윈도우 적용
        # (B, T, M) → (B, n_frames, M, K)
        x["signal"] = self.window(x["signal"])

        # GCC-PHAT 계산: 모든 마이크 쌍에 대해 상호상관 피처 계산
        # (B, n_frames, M, K) → (B, n_frames, M, M, nb_gcc_bins)
        # unsqueeze(-1): 채널 차원 추가 → (B, n_frames, M, M, nb_gcc_bins, 1)
        x["signal"] = self.feature_extractor(x["signal"]).unsqueeze(-1)

        # 지정한 방식으로 마이크 쌍을 서브샘플링
        # feature_pairs: (B, n_frames, n_pairs, nb_gcc_bins, 1)
        # idxs: (B, n_pairs, 2) — 선택된 마이크 쌍 인덱스
        feature_pairs, idxs = self._sample_pairs(
            x,
            self.n_mic_pairs,
            self.mic_pair_sampling_mode,
        )

        # 시간 프레임과 마이크 쌍 차원 교환: NeuralSrp forward 입력 형식 맞춤
        # (B, n_frames, n_pairs, F, C) → (B, n_pairs, n_frames, F, C)
        x["signal"] = feature_pairs.transpose(1, 2)

        # 선택된 쌍 인덱스에 따라 메타데이터(마이크 위치)를 준비
        # 반환: (B, n_pairs, n_metadata) — metadata_type에 따라 내용이 달라짐
        x["mic_pos"] = prepare_mic_pos(x["mic_pos"], idxs, mode=self.metadata_type)

        return x

    def _sample_pairs(self, x, n_pairs, mode="random"):
        """
        배치 피처 행렬에서 마이크 쌍을 서브샘플링.

        Args:
            x (dict): x["signal"].shape == (batch_size, n_frames, n_mics, n_mics, n_feature, n_channels)
                      x["mic_pos"]: 마이크 위치 정보
            n_pairs: 샘플링할 GCC 쌍의 수
            mode: "random", "first", "all" 중 하나

        Returns:
            (batch_size, n_frames, n_pairs, n_features, n_channels)
        """
        # GCC 피처 행렬: (B, n_frames, M, M, F, C)
        feature_matrix = x["signal"]
        # 마이크 위치: (B, M, 3)
        mic_pos = x["mic_pos"]

        assert mode in ["all", "random", "first", "distinct_angles"]
        # shape 언팩: 배치, 프레임, 마이크, 마이크, 피처, 채널
        batch_size, n_frames, n_mics, _, n_features, n_channels = feature_matrix.shape

        # 선택할 쌍 인덱스를 담을 빈 텐서 초기화: (B, n_pairs, 2)
        idx_pairs = torch.zeros(
            (batch_size, n_pairs, 2),
            dtype=torch.long,
            device=feature_matrix.device,
        )

        # 중복 없는 모든 마이크 쌍 인덱스 생성: (n_all_pairs, 2)
        # 예: M=4이면 [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] 6쌍
        idx_all_pairs = get_all_pairs(n_mics, device=feature_matrix.device)
        n_all_pairs = idx_all_pairs.shape[0]

        if mode == "first":
            # "first": 마이크 0을 기준으로 나머지 마이크와의 쌍만 선택
            # 예: M=4 → [(0,1),(0,2),(0,3)]
            idx_pairs = torch.tensor(
                [[0, i] for i in range(1, n_mics)],
                dtype=torch.long,
                device=feature_matrix.device,
            )
        elif mode == "all":
            # "all": 모든 마이크 쌍 사용
            idx_pairs = idx_all_pairs
        elif mode == "random":
            # "random": 전체 쌍 중 n_pairs개를 무작위 비복원 샘플링
            perm = torch.randperm(n_all_pairs)[:n_pairs]
            idx_pairs = idx_all_pairs[perm]
        elif mode == "distinct_angles":
            # "distinct_angles": 다양한 방향 커버리지를 위한 쌍 선택
            # 배치 내 모든 샘플이 동일한 마이크 배열을 가정하여 첫 번째 샘플 위치만 사용
            idx_pairs = select_pairs(mic_pos[0])
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")

        # 배치 차원 추가 후 배치 크기만큼 반복: (n_pairs, 2) → (B, n_pairs, 2)
        idx_pairs = idx_pairs.unsqueeze(0).expand(batch_size, -1, -1)

        # 실제 선택된 쌍 수 갱신 (mode="all"/"first"일 때 n_pairs와 다를 수 있음)
        n_pairs = idx_pairs.shape[1]
        # 선택된 쌍의 GCC 피처를 담을 출력 텐서 초기화
        # shape: (B, n_frames, n_pairs, n_features, n_channels)
        pairs = torch.zeros(
            (batch_size, n_frames, n_pairs, n_features, n_channels),
            device=feature_matrix.device,
            dtype=feature_matrix.dtype,
        )

        for i in range(batch_size):
            # i번째 배치에서 선택된 쌍의 피처를 인덱싱하여 복사
            # idx_pairs[i, :, 0]: 첫 번째 마이크 인덱스 (n_pairs,)
            # idx_pairs[i, :, 1]: 두 번째 마이크 인덱스 (n_pairs,)
            # feature_matrix[i]: (n_frames, M, M, F, C) → 인덱싱 → (n_frames, n_pairs, F, C)
            pairs[i] = feature_matrix[i, :, idx_pairs[i, :, 0], idx_pairs[i, :, 1]]

        return pairs, idx_pairs
        # pairs: (B, n_frames, n_pairs, n_features, n_channels)
        # idx_pairs: (B, n_pairs, 2)
