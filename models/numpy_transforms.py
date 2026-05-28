import math
from typing import Any  # 현재 파일 내에서 사용되지 않는 dead import
import numpy as np
import librosa


def n_choose_r(n, r):
    # 이항계수 C(n, r) = n! / (r! * (n-r)!) 를 정수 나눗셈으로 계산
    # 마이크 쌍의 총 수를 구할 때 사용 (예: 4채널 → C(4,2) = 6쌍)
    return math.factorial(n) // math.factorial(r) // math.factorial(n - r)


class GccExtractor:
    def __init__(self, params, gcc_mode="first"):
        """
        GCC-PHAT 특징 추출기를 초기화.

        :param params: 설정 파라미터 딕셔너리.
                       필수 키: "fs" (샘플링 주파수), "hop_rate", "win_size",
                                "nb_gcc_bins", "dataset.max_audio_len_s"
        :param gcc_mode: 마이크 쌍 구성 방식.
                         "first" → 채널 0을 기준으로 나머지 채널과 쌍 구성 (nb_ch-1개 쌍),
                         그 외   → 모든 채널 조합 사용 (C(nb_ch, 2)개 쌍)
        """

        self._fs = params["fs"]                          # 샘플링 주파수 (Hz), 예: 24000
        # TODO: move hop rate and win size out of neural_srp
        # hop_rate * win_size / fs = hop_len_samples / fs → 초 단위 hop 길이
        # 이후 _hop_len 계산 시 fs를 다시 곱하므로 fs가 약분됨 (중간 변수가 불필요할 수 있음)
        self._hop_len_s = params["hop_rate"] * params["win_size"] / params["fs"]
        self._nb_bins = params["nb_gcc_bins"]            # GCC 출력 lag bin 수 (최종 특징 차원), 예: 64
        self._gcc_mode = gcc_mode                        # 마이크 쌍 구성 방식: "first"=채널0 기준, 그 외=전체 조합

        self._hop_len = int(self._fs * self._hop_len_s)  # hop 길이를 샘플 수로 변환: fs * hop_len_s
        self._win_len = 2 * self._hop_len                # STFT 윈도우 길이 = hop의 2배 (50% overlap 고정)
        self._nfft = _next_greater_power_of_2(self._win_len)  # FFT 크기: win_len 이상의 최소 2의 거듭제곱

        self._eps = 1e-8  # 수치 안정용 epsilon (현재 주석 처리된 PHAT weighting에서 사용 예정이었으나 미사용)

        # Max audio length in samples
        audio_max_len_samples = params["dataset"]["max_audio_len_s"] * self._fs  # 최대 오디오 길이(초) → 샘플 수
        self._max_feat_frames = int(
            np.ceil(audio_max_len_samples / float(self._hop_len))  # 최대 오디오에서 생성될 수 있는 최대 STFT 프레임 수
        )

    def _spectrogram(self, audio_input):
        # audio_input shape: (N_samples, nb_ch)
        nb_ch = audio_input.shape[1]  # 마이크(채널) 수 추출

        spectra = []  # 채널별 STFT 결과를 담을 리스트
        for ch_cnt in range(nb_ch):  # 각 채널에 대해 순차적으로 STFT 계산
            stft_ch = librosa.stft(
                np.asfortranarray(audio_input[:, ch_cnt]),  # column slicing 후 Fortran-contiguous로 변환 (librosa 내부 최적화)
                n_fft=self._nfft,           # FFT 크기 (2의 거듭제곱)
                hop_length=self._hop_len,   # 프레임 간 이동 샘플 수
                win_length=self._win_len,   # 실제 분석 윈도우 길이 (nfft > win_len 이면 zero-padding 발생)
                window="hann",              # Hann 윈도우: spectral leakage 최소화
            )
            # stft_ch shape: (nfft//2 + 1, nb_frames_full) — 복소수 스펙트럼
            spectra.append(stft_ch[:, : self._max_feat_frames])  # 프레임 수를 최대값으로 clip하여 shape 균일화
        # spectra: nb_ch개의 (nb_stft_bins, nb_frames) 배열 리스트
        spectra = np.stack(spectra, axis=-1).transpose(1, 0, 2)
        # np.stack(..., axis=-1) shape: (nb_stft_bins, nb_frames, nb_ch)
        # .transpose(1, 0, 2)  shape: (nb_frames, nb_stft_bins, nb_ch)

        return spectra  # shape: (nb_frames, nb_stft_bins, nb_ch), dtype: complex

    def _get_gcc(self, linear_spectra):
        # linear_spectra shape: (nb_frames, nb_stft_bins, nb_ch)
        nb_frames, nb_stft_bins, nb_ch = linear_spectra.shape  # 입력 shape 언패킹

        if self._gcc_mode == "first":  # Use first channel as reference
            n_output_channels = nb_ch - 1      # 채널 0을 기준으로 나머지와 쌍 구성 → nb_ch-1개
        else:  # All combinations
            n_output_channels = n_choose_r(nb_ch, 2)  # 모든 마이크 쌍 조합 수: C(nb_ch, 2)

        gcc_feat = np.zeros((nb_frames, self._nb_bins, n_output_channels))  # 출력 버퍼 초기화, shape: (nb_frames, nb_bins, n_output_channels)
        mic_pair_idxs = []  # 마이크 쌍 인덱스 기록용 (수집 후 반환하거나 사용하지 않는 dead variable)

        cnt = 0  # 출력 채널(마이크 쌍) 카운터
        for m in range(nb_ch):          # 기준 마이크 인덱스
            for n in range(m + 1, nb_ch):  # m보다 큰 인덱스만 순회 → 중복 쌍 방지 (상삼각 조합)
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                # Cross-power spectrum: R[f] = X_m*(f) * X_n(f)
                # shape: (nb_frames, nb_stft_bins), dtype: complex
                # Compute the cross-power spectrum
                # R /= np.abs(R) + self._eps  # PHAT weighting: R_hat[f] = R[f] / |R[f]| (현재 비활성화)
                cc = np.fft.irfft(np.exp(1.0j * np.angle(R))) # Compute the GCC-PHAT
                # np.angle(R): R의 위상(phase)만 추출, shape: (nb_frames, nb_stft_bins)
                # np.exp(1j * angle): 단위 복소수로 변환 = R / |R| (PHAT weighting과 수학적 동치)
                # irfft(...): 역FFT로 GCC-PHAT 계산, shape: (nb_frames, nfft)
                cc = np.concatenate(
                    [  # Only keep central self._nb_bins
                        # IFFT 출력에서 음의 lag는 뒷부분, 양의 lag는 앞부분에 위치
                        # 뒷부분(-nb_bins//2:)과 앞부분(:nb_bins//2)을 이어 붙여 zero-lag를 중앙으로 이동 (circular shift)
                        # 주의: nb_bins가 홀수이면 concat 결과가 nb_bins-1이 되어 shape mismatch 발생 가능
                        cc[:, -self._nb_bins // 2 :],  # 음의 lag 영역: shape (nb_frames, nb_bins//2)
                        cc[:, : self._nb_bins // 2],   # 양의 lag 영역: shape (nb_frames, nb_bins//2)
                    ],
                    axis=-1,
                )
                # cc shape: (nb_frames, nb_bins) — 중앙 lag bin만 추출된 GCC-PHAT
                gcc_feat[:, :, cnt] = cc  # 현재 마이크 쌍의 결과를 출력 버퍼에 저장
                cnt += 1                  # 다음 마이크 쌍 슬롯으로 이동
                mic_pair_idxs.append([m, n])  # 마이크 쌍 인덱스 기록 (이후 미사용 dead variable)
            if self._gcc_mode == "first":
                break  # m=0 (첫 번째 채널) 루프 한 번만 실행하고 탈출

        gcc_feat = gcc_feat.transpose((0, 2, 1))
        # (nb_frames, nb_bins, n_output_channels) → (nb_frames, n_output_channels, nb_bins)

        return gcc_feat  # shape: (nb_frames, n_output_channels, nb_bins), dtype: float64

    def forward(self, audio_in, labels=None):
        # audio_in shape: (N_samples, nb_ch)
        spect = self._spectrogram(audio_in)   # STFT 계산 → shape: (nb_frames, nb_stft_bins, nb_ch)
        feat = self._get_gcc(spect)           # GCC-PHAT 계산 → shape: (nb_frames, n_output_channels, nb_bins)

        return feat, labels  # labels는 변환 없이 pass-through (transform chain에서 레이블 보존)

    def __call__(self, audio_in, labels=None):
        # 인스턴스를 함수처럼 호출할 수 있게 하는 매직 메서드 (extractor(audio, labels) 형태)
        return self.forward(audio_in, labels)


def _next_greater_power_of_2(x):
    # (x-1).bit_length() = ceil(log2(x)) → 2의 거듭제곱 중 x 이상의 최솟값 반환
    # 예: x=5 → (4).bit_length()=3 → 2^3=8 / x=8 → (7).bit_length()=3 → 2^3=8 (자기 자신)
    return 2 ** (x - 1).bit_length()


class WindowTargets:
    """윈도우 분절 변환기.
    윈도우 길이(K)와 윈도우 간 스텝을 지정하여 생성.
    선택적으로 길이 K의 벡터 또는 NumPy 윈도우 함수로 윈도우 형태를 지정 가능.

    TODO: Move this code to pytorch
    """

    def __init__(self, K, step):
        self.K = K      # 윈도우 크기 (샘플 수 단위)
        self.step = step  # 윈도우 간 hop 크기 (샘플 수 단위)

    def __call__(self, x, acoustic_scene):
        N_mics = x.shape[1]                        # 마이크 수 (이후 미사용 dead variable)
        N_dims = acoustic_scene["DOA"].shape[1]     # DOA 차원 수 (예: azimuth+elevation=2) (이후 미사용 dead variable)
        L = x.shape[0]                             # 오디오 총 샘플 수
        N_w = np.floor(L / self.step - self.K / self.step + 1).astype(int)
        # 총 윈도우 수: floor((L - K) / step) + 1 (이후 미사용 dead variable)

        if self.K > L:
            raise Exception(
                f"The window size can not be larger than the signal length ({L})"
            )
        elif self.step > L:
            raise Exception(
                f"The window step can not be larger than the signal length ({L})"
            )

        DOAw = to_frames(acoustic_scene["DOA"], self.K, self.step)
        # acoustic_scene["DOA"] shape: (L, N_dims)
        # DOAw shape: (n_frames, K, N_dims) — DOA를 윈도우 단위로 분절

        for i in np.flatnonzero(
            np.abs(np.diff(DOAw[..., 1], axis=1)).max(axis=1) > np.pi
        ):
            # DOAw[..., 1]: 마지막 축의 인덱스 1 = azimuth만 꺼냄, shape (n_frames, K)
            # np.diff(..., axis=1): 시간 축(axis=1) 방향으로 인접 샘플 간 차이 계산, shape (n_frames, K-1)
            # np.abs(...): 차이의 절댓값 → 방향 상관없이 크기만 봄
            # .max(axis=1): 각 프레임 내에서 가장 큰 차이값 하나만 추출, shape (n_frames,)
            # > np.pi ≈ 3.14: 차이가 π를 넘으면 경계 점프로 판단
            # np.flatnonzero(...): True인 인덱스만 꺼냄

            # Avoid jumping from -pi to pi in a window
            DOAw[i, DOAw[i, :, 1] < 0, 1] += 2 * np.pi
            # DOAw[i, :, 1] — i번 프레임의 azimuth 전체
            # DOAw[i, :, 1] < 0 — 음수인 위치를 True로
            # DOAw[i, <조건>, 1] — True인 위치의 값만 선택
            # += 2 * np.pi — 선택된 값에 2π 더하기
            # 해당 프레임 내 음수 방위각에 2π를 더해 [-π,π] 불연속을 [0,2π]로 통일
            # 단방향(음→양)만 처리하므로 반대 방향 불연속은 보정되지 않음 (np.unwrap 사용이 더 robust)
            
        DOAw = np.mean(DOAw, axis=1)
        # 각 윈도우 내 DOA를 시간 평균: (n_frames, K, N_dims) → (n_frames, N_dims)
        DOAw[DOAw[:, 1] > np.pi, 1] -= 2 * np.pi
        # 평균 후 방위각이 π를 초과하면 2π를 빼서 [-π, π] 범위로 복원
        acoustic_scene["DOAw"] = DOAw  # 윈도우 평균 DOA 저장, shape: (n_frames, N_dims)

        # Window the VAD if it exists
        if "vad" in acoustic_scene:
            acoustic_scene["vad"] = to_frames(acoustic_scene["vad"], self.K, self.step)
            # VAD 신호도 동일하게 윈도우 단위로 분절
            # acoustic_scene["vad"] shape: (L,) 또는 (L, n_sources) → (n_frames, K) 또는 (n_frames, K, n_sources)

        # Timestamp for each window
        acoustic_scene["tw"] = (
            np.arange(0, (L - self.K), self.step) / acoustic_scene["fs"]
            # 각 윈도우 시작 샘플 인덱스를 초 단위로 변환
            # np.arange(0, L-K, step): 각 윈도우의 시작 샘플 인덱스
            # 주의: 이 길이가 to_frames의 실제 n_frames와 off-by-one 차이가 발생할 수 있음
        )

        # Return the original signal
        return x, acoustic_scene  # x는 변경 없이 pass-through, acoustic_scene에 DOAw/tw 키 추가됨


def to_frames(x, frame_size, hop_size):
    """신호를 프레임 단위로 분절. 신호의 첫 번째 축(시간 축)을 기준으로 분절.

    Args:
        x (np.ndarray): 입력 신호.
        frame_size (int): 프레임 크기 (샘플 수).
        hop_size (int): 프레임 간 이동 간격 (샘플 수).
    Returns:
        np.ndarray: 분절된 신호, shape (n_frames, frame_size, ...)
    """

    x_shape = x.shape        # 입력 전체 shape 저장 (추후 출력 shape 구성에 사용)
    n_signal = x_shape[0]    # 시간 축(첫 번째 축) 길이

    n_frames = int(n_signal / hop_size - frame_size / hop_size)
    # 프레임 수: (n_signal - frame_size) / hop_size 와 수학적으로 동치
    # 부동소수점 나눗셈 후 int truncation → (n_signal - frame_size) // hop_size 로 변경하면 오차 제거 가능

    n_signal = n_frames * hop_size + frame_size
    # n_frames개의 프레임을 정확히 담는 데 필요한 최소 샘플 수 계산
    # (원래 n_signal 변수를 재사용하여 덮어씀)
    # Truncate the signal to fit an integer number of frames
    x = x[:n_signal]  # trailing 샘플 제거하여 프레임 수에 맞게 truncate

    out_shape = (n_frames, frame_size) + x_shape[1:]  
    # 출력 shape: (n_frames, frame_size, x_shape의 0번째 제외한 나머지 차원...) 
    # 예: (n_frames, K, N_dims)
    x_frames = np.zeros(out_shape, dtype=x.dtype)     # 출력 배열 0으로 초기화 (dtype 보존)

    for i in range(n_frames):
        x_frames[i] = x[i * hop_size : i * hop_size + frame_size]
        # i번째 프레임: 시작 샘플 i*hop_size, 끝 샘플 i*hop_size + frame_size - 1

    return x_frames  # shape: (n_frames, frame_size, ...), dtype: x.dtype
