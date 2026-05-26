import torch
import torch.nn as nn
import typing

from torchaudio.transforms import MelScale


class Dft(nn.Module):
    """입력 신호의 DFT 스펙트럼을 계산. phase_only=True이면 위상각만 반환."""

    def __init__(self, n_dft: int = 512, phase_only: bool = False):
        super().__init__()  

        # n_dft*2 이상인 최소 2의 거듭제곱으로 FFT 크기 설정 (zero-padding으로 주파수 해상도 향상)
        self.n_dft = _next_greater_power_of_2(n_dft*2)
        # True이면 복소수 FFT 대신 위상(angle)만 반환
        self.phase_only = phase_only

    def forward(self, x):
        batch_size, n_frames, n_mics, frame_size = x.shape  # shape: [B, F, M, K]

        # 실수 입력에 rfft 적용 — n_dft//2+1 개의 복소수 주파수 빈 반환
        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)    # shape: [B, F, M, n_dft//2+1], dtype: complex

        if self.phase_only:
            # 복소수 스펙트럼의 위상각(radian)만 추출 — angle = atan2(imag, real)
            x_fft = torch.angle(x_fft)  # shape: [B, F, M, n_dft//2+1], dtype: float

        return x_fft  # shape: [B, F, M, n_dft//2+1]


class MelSpectra(nn.Module):
    """입력 신호의 Mel 스케일 위상 스펙트럼을 계산"""

    def __init__(self, sample_rate: int = 16000, n_dft: int = 512,
                 f_min: float = 0.0, f_max: typing.Optional[float] = None,
                 n_mels: int = 128, phase_only: bool = False):
        super().__init__()  # nn.Module 초기화

        self.sample_rate = sample_rate  # 샘플링 주파수 (Hz)
        # n_dft 이상인 최소 2의 거듭제곱으로 FFT 크기 설정
        self.n_dft = _next_greater_power_of_2(n_dft)
        self.f_min = f_min          # Mel 필터뱅크 하한 주파수 (Hz)
        self.f_max = f_max          # Mel 필터뱅크 상한 주파수 (Hz), None이면 sample_rate/2
        self.n_mels = n_mels        # Mel 필터 개수
        self.phase_only = phase_only  # True이면 위상만 반환

        # torchaudio MelScale 필터뱅크 변환 객체 생성
        # n_stft = n_dft//2+1: rfft 출력의 주파수 빈 수
        self.mel_scale_transform = MelScale(n_mels,
                                            sample_rate,
                                            f_min,
                                            f_max,
                                            n_stft=self.n_dft // 2 + 1)

    def forward(self, x):
        batch_size, n_frames, n_mics, frame_size = x.shape  # shape: [B, F, M, K]

        # 실수 FFT 적용 — 각 프레임의 주파수 스펙트럼 계산
        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)    # shape: [B, F, M, n_dft//2+1], dtype: complex

        # MelScale은 마지막 두 차원이 (주파수, 시간) 순서여야 하므로 축 변경
        x_fft = x_fft.permute(0, 2, 3, 1)  # shape: [B, M, n_dft//2+1, F]
        # Permute to have freq, time as last two dimensions, as required by
        # torchaudio.transforms.MelScale

        # 실수부에 Mel 필터뱅크 적용 — n_dft//2+1 주파수 빈을 n_mels 빈으로 선형 합산
        x_mel_real = self.mel_scale_transform(
            x_fft.real  # shape: [B, M, n_dft//2+1, F] → [B, M, n_mels, F]
        )
        # 허수부에 동일한 Mel 필터뱅크 적용
        x_mel_imag = self.mel_scale_transform(
            x_fft.imag  # shape: [B, M, n_dft//2+1, F] → [B, M, n_mels, F]
        )
        # 실수부와 허수부를 합쳐 복소수 Mel 스펙트럼 복원
        x_mel = torch.complex(x_mel_real, x_mel_imag)  # shape: [B, M, n_mels, F], dtype: complex

        # 원래 차원 순서로 복원 (batch, frames, mics, mels)
        x_mel = x_mel.permute(0, 3, 1, 2)  # shape: [B, F, M, n_mels]

        if self.phase_only:
            # 복소 Mel 스펙트럼에서 위상각만 추출
            x_mel = torch.angle(x_mel)  # shape: [B, F, M, n_mels], dtype: float

        return x_mel  # shape: [B, F, M, n_mels]


class MelCrossSpectra(MelSpectra):
    """마이크 쌍 간의 Mel 스케일 교차 스펙트럼을 계산.

    MelSpectra로 각 마이크의 복소수 Mel 스펙트럼을 구한 뒤,
    모든 마이크 쌍 (n, m)에 대해 교차 스펙트럼 X_n(f) * conj(X_m(f))를 계산.
    교차 스펙트럼의 위상은 두 마이크 간 시간 지연(TDOA) 정보를 담고 있으며,
    음원 위치 추정(SRP) 모델의 공간 피처로 사용.

    입력: [B, F, M, K]
    출력: [B, F, M, M, n_mels] (dtype: complex64)
    """

    def __init__(self, sample_rate: int = 16000, n_dft: int = 512,
                 f_min: float = 0.0, f_max: typing.Optional[float] = None,
                 n_mels: int = 128, phase_only: bool = False):
        # 부모 클래스 MelSpectra의 초기화 위임
        super().__init__(sample_rate, n_dft, f_min, f_max, n_mels, phase_only)

    def forward(self, x):
        # 부모 클래스의 forward로 Mel 스케일 스펙트럼 계산
        x_mel = super().forward(x)  # shape: [B, F, M, n_mels], dtype: complex

        batch_size, n_frames, n_mics, n_mels = x_mel.shape  # shape: [B, F, M, n_mels]

        # 마이크 쌍 간의 교차 스펙트럼을 저장할 빈 텐서 초기화
        # cross_spectra[b, f, n, m, k] = x_mel[b, f, n, k] * conj(x_mel[b, f, m, k])
        cross_spectra = torch.zeros(
            batch_size, n_frames, n_mics, n_mics, n_mels,  # shape: [B, F, M, M, n_mels]
            device=x.device, dtype=torch.complex64          # 복소수 64비트 (실수32 + 허수32)
        )

        for n in range(n_mics):  # n번째 마이크를 기준 채널로 순회
            # n번 마이크 스펙트럼과 전체 마이크 스펙트럼 켤레의 원소별 곱 = 교차 스펙트럼
            # unsqueeze(2): [B, F, n_mels] → [B, F, 1, n_mels] (브로드캐스트용 마이크 차원 추가)
            # n번째 마이크 스펙트럼 * 모든 마이크 스펙트럼(conjugate)
            # ex) n=0: cross_spectra[:,:, 0, :, :] = [X_0·X_0*, X_0·X_1*, X_0·X_2*]
            # cross_spectra[:, :, n, m, :]는 n번 마이크와 m번 마이크의 교차 스펙트럼이 채워진 M×M 행렬
            # 대각선(n==m)은 자기 자신과의 곱이므로 파워 스펙트럼, 나머지는 TDOA 위상 정보
            cross_spectra[:, :, n] = x_mel[:, :, n].unsqueeze(2) * x_mel.conj()  # shape: [B, F, M, n_mels]

        return cross_spectra  # shape: [B, F, M, M, n_mels], dtype: complex64


class GCC(nn.Module):
    """입력 신호의 일반화 상호 상관(GCC)을 계산.
    레이어 생성자에서 신호 수(N)와 윈도우 길이(K)를 지정.
    tau_max로 GCC의 중앙 부분만 출력하거나, transform='PHAT'으로 PHAT 변환 사용 가능.
    """

    def __init__(self, K, tau_max=None, transform=None, concat_bins=False, center=False, abs=True):
        # transform 인자 유효성 검사 — 현재 'phat'만 지원
        assert (
            transform is None or transform == "phat"
        ), "Only the 'PHAT' transform is implemented"
        # tau_max는 프레임 크기의 절반을 초과할 수 없음
        assert tau_max is None or tau_max <= K // 2
        super().__init__()  # nn.Module 초기화

        self.K = K  # 프레임(윈도우) 크기 (샘플 수)
        # FFT 크기: K 이상의 최소 2의 거듭제곱 (zero-padding 효과)
        self.n_dft = _next_greater_power_of_2(K)
        # tau_max 미지정 시 K//2 (최대 가능한 시간 지연)
        self.tau_max     = tau_max if tau_max is not None else K // 2
        # 'phat': PHAT(위상 변환) 가중치 적용 여부
        self.transform = transform
        # True이면 GCC 값과 시간 지연 빈 인덱스를 채널로 연결
        self.concat_bins = concat_bins
        # True이면 GCC를 영-중심(zero-centered)으로 재배열
        self.center = center
        # True이면 최종 GCC의 절댓값 반환
        self.abs = abs

    def forward(self, x, tau_max=None):
        """입력 신호의 GCC를 계산.
        x: [batch_size, n_frames, n_mics, frame_size] 형태의 텐서.
           K는 시간 샘플 수, N은 마이크 채널 수.
        tau_max (선택): 출력할 최대 시간 지연 (샘플 단위).

        반환값: [batch_size, N, N, 2*tau_max+1] 형태의 텐서 — 각 마이크 쌍의 GCC
        """

        batch_size, n_frames, n_mics, frame_size = x.shape  # shape: [B, F, M, K]

        # forward 호출 시 tau_max를 별도 지정하지 않으면 self.tau_max 사용
        if tau_max is None:
            tau_max = self.tau_max

        # 실수 FFT 적용 — 각 프레임을 주파수 도메인으로 변환
        x_fft = torch.fft.rfft(x, n=self.n_dft, dim=-1)  # shape: [B, F, M, n_dft//2+1], dtype: complex

        if self.transform == "phat":
            # PHAT 가중치: 각 주파수 빈의 크기를 1로 정규화 — 위상 정보만 보존
            # 수식: X_phat(f) = X(f) / |X(f)|, 1e-8은 분모=0 방지 epsilon
            x_fft /= x_fft.abs() + 1e-8  # To avoid numerical issues

        # GCC 결과를 저장할 빈 텐서 생성 — 각 마이크 쌍(n, m)에 대해 2*tau_max 개의 시간 지연 빈
        gcc = torch.empty(
            [batch_size, n_frames, n_mics, n_mics, 2 * tau_max],  # shape: [B, F, M, M, 2*tau_max]
            device=x.device,
        )

        for n in range(n_mics):  # n번째 마이크를 기준 채널로 순회
            # n번 마이크 스펙트럼과 전체 마이크 스펙트럼 켤레의 원소별 곱 = 교차 전력 스펙트럼
            # unsqueeze(2): [B, F, n_dft//2+1] → [B, F, 1, n_dft//2+1] (브로드캐스트용)
            gcc_fft_batch = x_fft[:, :, n].unsqueeze(2) * x_fft.conj()  # shape: [B, F, M, n_dft//2+1]

            # 역 실수 FFT로 교차 전력 스펙트럼을 시간 도메인의 GCC로 변환
            gcc_batch = torch.fft.irfft(gcc_fft_batch, dim=-1)  # shape: [B, F, M, n_dft]

            if self.center:
                # center=True: 음의 시간 지연(-tau_max~-1)을 앞에, 양의 지연(0~tau_max-1)을 뒤에 배치
                # irfft 결과의 마지막 tau_max 샘플 = 음의 지연 (순환 대칭 특성)
                gcc[:, :, n, :, :tau_max] = gcc_batch[..., -tau_max:]   # 음의 지연 구간
                gcc[:, :, n, :, tau_max:] = gcc_batch[..., :tau_max]    # 양의 지연 구간
            else:
                # center=False: 양의 지연(0~tau_max-1)을 앞에, 음의 지연을 뒤에 배치
                gcc[:, :, n, :, :tau_max] = gcc_batch[..., :tau_max]    # 양의 지연 구간
                gcc[:, :, n, :, -tau_max:] = gcc_batch[..., -tau_max:]  # 음의 지연 구간

        if self.concat_bins:
            # GCC 값에 시간 지연 빈 인덱스(샘플 단위)를 마지막 채널로 연결
            bins = get_gcc_bins(tau_max, x.device,
                                center=self.center)[:-1]  # 출력 shape 일치를 위해 마지막 빈 제거
            # gcc와 동일 shape의 텐서에 bins 값을 브로드캐스트
            bins = torch.ones_like(gcc)*bins              # shape: [B, F, M, M, 2*tau_max]
            # GCC와 bins를 마지막 차원 뒤에 새 차원으로 쌓아 연결
            gcc = torch.stack([gcc, bins], dim=-1)         # shape: [B, F, M, M, 2*tau_max, 2]

        if self.abs:
            # 절댓값 반환: GCC의 크기(amplitude)만 사용
            gcc = gcc.abs()

        return gcc  # shape: [B, F, M, M, 2*tau_max] 또는 [B, F, M, M, 2*tau_max, 2]


def _next_greater_power_of_2(x):
    # x 이상인 최소 2의 거듭제곱 반환
    # 예: x=5 → (5-1).bit_length()=3 → 2^3=8
    return 2 ** (x - 1).bit_length()


class Window(nn.Module):
    """윈도우 변환.
    프레임 크기, 윈도우 간격(hop), 선택적 윈도우 종류(rectangular 또는 hanning)를 지정하여 생성한다.

    (batch_size, n_frames, K, n_mics) 형태의 텐서를 반환한다.
    """

    def __init__(self, frame_size, hop_size, window=None):
        self.frame_size = frame_size    # 각 프레임의 샘플 수 (K)
        self.hop_size = hop_size        # 프레임 간 이동 샘플 수
        self.window_name = window       # 윈도우 종류: None(rectangular) 또는 'hann'
        self.window = None              # 실제 윈도우 텐서 — forward에서 지연 초기화

        super().__init__()

    def forward(self, x):
        batch_size, n_signal, n_mics = x.shape  # shape: [B, T, M]

        # 프레임 크기가 신호 길이보다 크면 분석 불가
        if self.frame_size > n_signal:
            raise Exception(
                f"The window size can not be larger than the signal length ({n_signal})"
            )
        # hop 크기가 신호 길이보다 크면 프레임 생성 불가
        elif self.hop_size > n_signal:
            raise Exception(
                f"The window step can not be larger than the signal length ({n_signal})"
            )

        # 윈도우 텐서가 아직 생성되지 않은 경우 지연 초기화 (디바이스·dtype 맞춤)
        if self.window is None:
            if self.window_name is None:
                # rectangular 윈도우: 모든 계수가 1 (곱해도 신호 변화 없음)
                self.window = torch.ones((self.frame_size, n_mics), dtype=x.dtype, device=x.device)  # shape: [K, M]
            elif self.window_name == "hann":
                # Hann 윈도우: 양 끝에서 0으로 감소하는 코사인 윈도우 — 스펙트럼 누설 억제
                self.window = torch.hann_window(self.frame_size, dtype=x.dtype, device=x.device)     # shape: [K]

        # 정수 프레임 수 계산: floor(n_signal/hop_size) - floor(frame_size/hop_size)
        n_frames = int(n_signal / self.hop_size - self.frame_size / self.hop_size)
        # n_frames개의 프레임이 정확히 들어맞도록 신호 길이 재계산
        n_signal = n_frames * self.hop_size + self.frame_size
        # 재계산된 길이로 신호를 잘라 정수 프레임 수에 맞게 맞춤
        x = x[:, :n_signal]  # shape: [B, n_signal, M]

        # (batch_size, n_frames, frame_size, n_mics) 형태의 프레임 텐서 생성
        x_frames = torch.zeros(
            (batch_size, n_frames, self.frame_size, n_mics), dtype=x.dtype, device=x.device
        )  # shape: [B, n_frames, K, M]

        for i in range(n_frames):
            # hop_size * i 위치부터 frame_size 샘플을 슬라이싱하여 i번째 프레임에 저장
            x_frames[:, i] = x[:, i * self.hop_size : i * self.hop_size + self.frame_size]  # shape: [B, K, M]

        # 윈도우 함수를 프레임 텐서에 원소별 곱 적용
        # unsqueeze로 차원 맞춤: [K, M] 또는 [K] → [1, 1, K, 1]
        x_frames =  x_frames*self.window.unsqueeze(0).unsqueeze(1).unsqueeze(3)  # shape: [B, n_frames, K, M]

        # 마지막 두 차원 교환: (frame_size, n_mics) → (n_mics, frame_size)
        return x_frames.transpose(2, 3)  # shape: [B, n_frames, M, K]


def get_gcc_bins(tau_max, device, center=True):
    # center=False는 미구현 — 현재 center=True만 지원
    if not center:
        raise NotImplementedError("Only center=True is implemented")

    # (미사용) 아래 코드에서 덮어쓰므로 실질적으로 사용되지 않음
    gcc_bins = torch.cat([
        torch.arange(0, tau_max, device=device),
        torch.arange(-tau_max, 0, device=device)
    ])

    # 영-중심 배열: [-tau_max, ..., -1, 0, 1, ..., tau_max] 순서의 시간 지연 인덱스 생성
    gcc_bins = torch.zeros(2*tau_max + 1, device=device)   # shape: [2*tau_max+1]
    # 앞 절반: -tau_max ~ -1 (내림차순 음수 지연)
    gcc_bins[0:tau_max] = - torch.arange(tau_max, 0, -1)   # [-tau_max, -(tau_max-1), ..., -1]
    # 뒷 절반: 0 ~ tau_max (오름차순 양수 지연)
    gcc_bins[tau_max:] = torch.arange(0, tau_max + 1)      # [0, 1, ..., tau_max]

    return gcc_bins  # shape: [2*tau_max+1], 값: 시간 지연 (샘플 단위)


def compute_tau_max(mic_pos, c, fs):
    # 마이크 개수 확인
    N = mic_pos.shape[0]

    # 모든 마이크 쌍 사이의 유클리드 거리 중 최댓값 계산
    # 수식: dist_max = max_{n,m} ||mic_pos[n] - mic_pos[m]||_2
    dist_max = max(
        [
            max([
                torch.linalg.norm(
                    mic_pos[n, :] - mic_pos[m, :])  # n번과 m번 마이크 사이의 유클리드 거리
                for m in range(N)
            ])
            for n in range(N)
        ]
    )  # dist_max: 마이크 배열에서 가장 먼 두 마이크 사이의 거리 (미터 단위)

    # 최대 시간 지연: tau_max = ceil(dist_max / c * fs)
    # 물리적 의미: 음속 c(m/s)로 dist_max(m)를 이동하는 시간을 샘플로 변환
    return int(torch.ceil(dist_max / c * fs))  # 반환값: 정수 샘플 수
