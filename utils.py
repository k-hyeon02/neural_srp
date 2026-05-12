"""
    Utils functions to deal with spherical coordinates in Pytorch.

    File name: utils.py
    Author: David Diaz-Guerra
    Date creation: 05/2020
    Python Version: 3.8
    Pytorch Version: 1.4.0
"""

import json
import matplotlib.pyplot as plt
import math
import os
import numpy as np
import torch


def stack_dicts(dicts_list):
    """여러 개의 딕셔너리를 같은 key별로 쌓아 하나의 딕셔너리로 만드는 함수"""

    # 결과를 저장할 빈 딕셔너리를 생성
    stacked_dict = {}
    # 첫 번째 딕셔너리의 key들을 기준으로 모든 딕셔너리를 순회
    for key in dicts_list[0].keys():
        # 각 딕셔너리에서 같은 key의 값을 꺼내 NumPy 배열로 stack
        stacked_dict[key] = np.stack([d[key] for d in dicts_list])
    # key별로 stack된 딕셔너리를 반환
    return stacked_dict


def cart2sph(cart, include_r=False):
    """PyTorch 텐서의 직교좌표계 좌표를 구면좌표계 좌표로 변환하는 함수
    Each row contains one point in format (x, y, x) or (elevation, azimuth, radius),
    where the radius is optional according to the include_r argument.
    """

    # 각 점의 원점으로부터의 거리 r을 계산
    r = torch.sqrt(torch.sum(torch.pow(cart, 2), dim=-1))
    # z축 기준으로 내려오는 극각 theta를 계산
    theta = torch.acos(cart[..., 2] / r)
    # x-y 평면에서의 방위각 phi를 계산
    phi = torch.atan2(cart[..., 1], cart[..., 0])
    # 반지름 r까지 반환해야 하는지 확인
    if include_r:
        # theta, phi, r을 마지막 차원에 묶어 구면좌표 텐서 생성
        sph = torch.stack((theta, phi, r), dim=-1)
    # 반지름 r을 반환하지 않는 경우
    else:
        # theta와 phi만 마지막 차원에 묶어 구면좌표 텐서 생성
        sph = torch.stack((theta, phi), dim=-1)
    # 변환된 구면좌표 텐서를 반환
    return sph


# NumPy 배열의 직교좌표계 좌표를 구면좌표계 좌표로 변환하는 함수
def cart2sph_np(cart, include_r=True):
    # x와 y 성분의 제곱합을 계산
    xy2 = cart[..., 0]**2 + cart[..., 1]**2
    # 입력 배열과 같은 shape의 결과 배열을 0으로 초기화
    sph = np.zeros_like(cart)
    # 반지름 r을 계산해 첫 번째 성분에 저장
    sph[..., 0] = np.sqrt(xy2 + cart[..., 2]**2)
    # z축 기준으로 내려오는 elevation/theta 각도를 계산해 두 번째 성분에 저장
    sph[..., 1] = np.arctan2(np.sqrt(xy2), cart[..., 2])
    # x-y 평면에서의 방위각 phi를 계산해 세 번째 성분에 저장
    sph[..., 2] = np.arctan2(cart[..., 1], cart[..., 0])
    
    # 반지름 r까지 포함해서 반환해야 하는지 확인
    if include_r:
        # r, theta, phi가 모두 포함된 구면좌표 배열을 반환
        return sph
    # 반지름 r을 제외하고 반환하는 경우
    else:
        # theta와 phi 성분만 반환
        return sph[..., 1:]


def sph2cart(sph):
    """PyTorch 텐서의 구면좌표계 좌표를 직교좌표계 좌표로 변환하는 함수
    Each row contains one point in format (elevation, azimuth, radius),
    where the radius is supposed to be 1 if it is not included.
    """

    # 입력에 theta와 phi만 있으면 반지름 1을 추가해 (theta, phi, r) 형태로 맞춤
    if sph.shape[-1] == 2: sph = torch.cat((sph, torch.ones_like(sph[..., 0]).unsqueeze(-1)), dim=-1)
    # 구면좌표 공식으로 x 좌표를 계산
    x = sph[..., 2] * torch.sin(sph[..., 0]) * torch.cos(sph[..., 1])
    # 구면좌표 공식으로 y 좌표를 계산
    y = sph[..., 2] * torch.sin(sph[..., 0]) * torch.sin(sph[..., 1])
    # 구면좌표 공식으로 z 좌표를 계산
    z = sph[..., 2] * torch.cos(sph[..., 0])
    # x, y, z를 마지막 차원에 묶어 직교좌표 텐서로 반환
    return torch.stack((x, y, z), dim=-1)


def acoustic_power(s):
    """음성 신호에서 무음 구간을 제외한 평균 음향 파워를 계산하는 함수"""

    # 무음 검출에 사용할 윈도우 크기를 512 샘플로 설정
    w = 512
    # 무음 검출에 사용할 윈도우 이동 간격을 256 샘플로 설정
    o = 256

    # Window the input signal
    # 입력 신호를 stride trick에 안전한 연속 메모리 배열로 변환
    s = np.ascontiguousarray(s)
    # 슬라이딩 윈도우로 볼 배열 shape를 정의
    sh = (s.size - w + 1, w)
    # 원본 stride를 두 번 반복해 2차원 윈도우 view의 stride 생성
    st = s.strides * 2
    # 별도 복사 없이 슬라이딩 윈도우 view를 만들고 o 간격으로 샘플링
    S = np.lib.stride_tricks.as_strided(s, strides=st, shape=sh)[0::o]

    # 각 윈도우의 평균 제곱값, 즉 파워를 계산
    window_power = np.mean(S ** 2, axis=-1)
    # 최대 윈도우 파워의 1%를 무음 판정 임계값으로 사용
    th = 0.01 * window_power.max()  # Threshold for silent detection
    # 임계값보다 큰 윈도우들만 골라 평균 파워를 반환
    return np.mean(window_power[np.nonzero(window_power > th)])


class Parameter:
    """고정값 또는 범위 기반 난수값을 다루기 위한 파라미터 클래스
    You can indicate a constant value or a random range in its constructor and then
    get a value acording to that with get_value(). It works with both scalars and vectors.
    """

    # 생성자에서 고정값 하나 또는 최솟값/최댓값 두 개를 받음
    def __init__(self, *args):
        # 인자가 하나면 항상 같은 값을 반환하는 고정 파라미터로 설정
        if len(args) == 1:
            # 이 파라미터가 난수 파라미터가 아님을 표시
            self.random = False
            # 입력값을 NumPy 배열로 저장
            self.value = np.array(args[0])
            # 고정 파라미터이므로 최솟값은 사용하지 않음
            self.min_value = None
            # 고정 파라미터이므로 최댓값은 사용하지 않음
            self.max_value = None
        # 인자가 두 개면 두 값 사이에서 난수를 뽑는 파라미터로 설정
        elif len(args) == 2:
            # 이 파라미터가 난수 파라미터임을 표시
            self. random = True
            # 난수 범위의 최솟값을 NumPy 배열로 저장
            self.min_value = np.array(args[0])
            # 난수 범위의 최댓값을 NumPy 배열로 저장
            self.max_value = np.array(args[1])
            # 난수 파라미터이므로 고정값은 사용하지 않음
            self.value = None
        # 인자 개수가 1개 또는 2개가 아니면 잘못된 사용
        else: 
            # 올바른 호출 방식이 아니므로 예외를 발생
            raise Exception('Parammeter must be called with one (value) or two (min and max value) array_like parammeters')

    # 현재 파라미터 설정에 따라 고정값 또는 난수값을 반환
    def get_value(self):
        # 난수 파라미터라면 최솟값과 최댓값 사이에서 균등분포 난수를 생성
        if self.random:
            # min_value shape에 맞춘 난수를 만들어 범위에 맞게 스케일링 (min_value ~ max_value)
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        # 고정 파라미터라면 저장된 값을 그대로 반환
        else:
            # 생성자에서 저장한 고정값을 반환
            return self.value


# 마이크 위치 점들과 마이크 쌍을 2D 그래프로 그리는 함수
def plot_pairs(points, pair_idxs, filename='', ax=None):
    # 외부에서 축 객체를 넘겨주지 않았는지 확인
    if ax is None:
        # 새 figure와 axis를 생성
        fig, ax = plt.subplots()

    # Plot the points
    # points 배열의 x, y 좌표를 산점도로 표시
    ax.scatter(points[:, 0], points[:, 1], label='# mics. = {}'.format(len(points)))
    # x축과 y축의 스케일을 같게 맞춥니다.
    ax.axis('equal')

    # Plot the pair vectors
    # pair_idxs에 정의된 마이크 쌍들을 선으로 연결
    for i, pair_idx in enumerate(pair_idxs):
        # legend에 사용할 label을 기본적으로 비움
        label = None
        # 첫 번째 선에만 전체 pair 개수 label을 붙힘
        if i == 0:
            # legend에 표시할 pair 개수 문자열을 만듦
            label = '# pairs = {}'.format(len(pair_idxs))
            
        # 현재 pair의 첫 번째 마이크 좌표를 가져옴
        mic_0 = points[pair_idx[0]]
        # 현재 pair의 두 번째 마이크 좌표를 가져옴
        mic_1 = points[pair_idx[1]]
        # 두 마이크를 선으로 연결해 시각화
        ax.plot([mic_0[0], mic_1[0]], [mic_0[1], mic_1[1]], 'r', label=label)

    ax.legend()
    # 저장 파일명이 전달되었는지 확인
    if filename:
        # 현재 figure를 지정된 파일명으로 저장
        plt.savefig(filename)

    # 사용자가 추가 조작할 수 있도록 axis 객체를 반환
    return ax


def plot_estimated_doa_from_acoustic_scene(acoustic_scene, output_path=None):
    """acoustic_scene 딕셔너리에서 DOA 정답과 예측값을 꺼내 플롯하는 함수
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    """

    # acoustic_scene에서 예측 DOA 값을 가져옴
    predicted_doa = acoustic_scene["DOAw_pred"]
    # acoustic_scene에서 정답 DOA 값을 가져옴
    target_doa = acoustic_scene["DOAw"]
    # acoustic_scene에서 VAD 정보를 가져옴
    vad = acoustic_scene["vad"]
    # acoustic_scene에서 원본 source signal을 가져옴
    source_signal = acoustic_scene["source_signal"]
    # 시간축의 마지막 값을 전체 duration으로 사용
    duration = acoustic_scene["tw"][-1]

    # 추출한 값들을 실제 플로팅 함수에 전달
    plot_estimated_doa(predicted_doa, target_doa, duration, source_signal, vad, output_path)


def plot_estimated_doa(predicted_doa, target_doa, duration=1,
                          source_signal=None, vad=None, output_path=None):
    """DOA 정답과 예측값을 시간축에 따라 시각화하는 함수
    The scene need to have the fields DOAw and DOAw_pred with the DOA groundtruth and the estimation.
    """

    # 새 matplotlib figure를 생성
    fig = plt.figure()

    # source_signal이 있으면 DOA 그래프 위에 source waveform도 함께 그림
    if source_signal is not None:
        # 7행 1열 grid를 만들어 위쪽에는 신호, 아래쪽에는 DOA를 배치
        gs = fig.add_gridspec(7, 1)
        # 아래쪽 큰 축과 위쪽 작은 축을 생성
        axs = fig.add_subplot(gs[1:,0]), fig.add_subplot(gs[0,0])
        # source_signal 길이에 맞춰 0부터 duration까지 시간축을 만듭니다.
        time_steps = np.linspace(0, duration, source_signal.shape[0])
        # 위쪽 축에 source signal waveform을 그림
        axs[1].plot(time_steps, source_signal)
        # x축 범위를 source signal 시간축 전체로 설정
        plt.xlim(time_steps[0], time_steps[-1])
        # 위쪽 waveform 축의 눈금과 라벨을 숨깁니다.
        plt.tick_params(axis='both', which='both', bottom=False, labelbottom=False, left=False, labelleft=False)
    # source_signal이 없으면 DOA만 그릴 단일 축을 사용
    else:
        # 단일 subplot 축을 생성
        axs = fig.subplots(1, 1)

    # target_doa 길이에 맞춰 0부터 duration까지 시간축을 만듭니다.
    time_steps = np.linspace(0, duration, target_doa.shape[0])

    # DOA의 각 차원에 붙일 라벨 이름을 정의
    labels = ["Azimuth", "Elevation"]
    # 각 DOA 차원을 그릴 색상을 정의
    colors = ["navy", "#83d44c"]

    # target_doa의 각 각도 차원에 대해 정답과 예측을 함께 그림
    for i in range(target_doa.shape[1]):
        # 현재 각도 차원의 정답 DOA를 radian에서 degree로 바꿔 실선으로 그림
        axs[0].plot(time_steps, target_doa[:, i] * 180/np.pi,
                      label=f"Target {labels[i]}", color=colors[i])
        # 현재 각도 차원의 예측 DOA를 radian에서 degree로 바꿔 점선으로 그림
        axs[0].plot(time_steps, predicted_doa[:, i] * 180/np.pi, '--',
                      label=f"Predicted {labels[i]}", color=colors[i])

    # 이후 추가 플롯이 기본 색상 순환을 다시 사용하도록 color cycle을 초기화
    plt.gca().set_prop_cycle(None)

    # DOA 축에 legend를 표시
    axs[0].legend(loc='best')
    # x축 라벨을 시간 단위로 설정
    axs[0].set_xlabel('time [s]')
    # y축 라벨을 DOA degree 단위로 설정
    axs[0].set_ylabel('DOA [º]')
    # x축 범위를 전체 시간 범위로 설정
    axs[0].set_xlim(time_steps[0], time_steps[-1])
    # y축 라벨을 오른쪽에 배치
    axs[0].yaxis.set_label_position("right")

    # vad가 있으면 무음 구간을 회색 배경으로 표시
    if vad is not None:
        # 각 시간 프레임의 평균 VAD가 2/3보다 작으면 무음으로 간주
        silences = vad.mean(axis=1) < 2/3
        # VAD 길이에 맞춰 새 시간축을 만듦
        time_steps = np.linspace(0, duration, silences.shape[0])
        # 무음으로 판정된 frame index들을 가져옴
        silences_idx = silences.nonzero()[0]
        # 연속 무음 구간의 시작점과 끝점을 저장할 리스트
        start, end = [], []
        # 무음 index들을 순회하며 연속 구간을 찾음
        for i in silences_idx:
            # 이전 index가 무음 index에 없으면 현재 index가 새 무음 구간의 시작
            if not i - 1 in silences_idx:
                # 현재 index를 무음 구간 시작 리스트에 추가
                start.append(i)
            # 다음 index가 무음 index에 없으면 현재 index가 무음 구간의 끝
            if not i + 1 in silences_idx:
                # 현재 index를 무음 구간 끝 리스트에 추가
                end.append(i)
        # 찾은 무음 시작/끝 쌍을 순회
        for s, e in zip(start, end):
            # 해당 시간 구간을 회색 반투명 영역으로 표시
            axs[0].axvspan((s-0.5)*time_steps[1], (e+0.5)*time_steps[1], facecolor='0.5', alpha=0.5)

    # output_path가 있으면 화면 표시 대신 파일로 저장
    if output_path is not None:
        # 여백을 타이트하게 맞춰 figure를 저장
        plt.savefig(output_path, bbox_inches='tight')
    # output_path가 없으면 화면에 그래프를 띄움
    else:
        # 현재 figure를 인터랙티브 창에 표시
        plt.show()


def dict_to_device(dict_of_tensors, device):
    """딕셔너리 안의 모든 텐서를 지정한 device로 이동시키는 함수
    Args:
        dict: A dictionary of tensors.
        device: The device to move the tensors to.
    """

    # 딕셔너리의 key와 value를 하나씩 순회
    for key, value in dict_of_tensors.items():
        # value가 PyTorch Tensor인지 확인
        if isinstance(value, torch.Tensor):
            # Tensor이면 지정된 device로 이동
            value = value.to(device)
        # value가 중첩 딕셔너리인지 확인
        elif isinstance(value, dict):
            # 중첩 딕셔너리라면 재귀적으로 내부 Tensor들을 device로 이동
            value = dict_to_device(value, device)
        # Tensor도 딕셔너리도 아니면 처리할 수 없는 타입
        else:
            # 지원하지 않는 타입이므로 예외를 발생시킵니다.
            raise ValueError('Value is nor a tensor or a dictionary.')
        # 변환된 value를 원래 key 위치에 다시 저장
        dict_of_tensors[key] = value

    # 모든 Tensor가 이동된 딕셔너리를 반환
    return dict_of_tensors


# 마이크 위치 점들과 마이크 쌍을 2D 그래프로 그리는 함수
def plot_pairs(points, pair_idxs, filename='', ax=None):
    # 외부에서 축 객체를 넘겨주지 않았는지 확인
    if ax is None:
        # 새 figure와 axis를 생성
        fig, ax = plt.subplots()

    # points 배열의 x, y 좌표를 산점도로 표시
    ax.scatter(points[:, 0], points[:, 1], label='# mics. = {}'.format(len(points)))
    # x축과 y축의 스케일을 같게 맞춤
    ax.axis('equal')

    # pair_idxs에 정의된 마이크 쌍들을 선으로 연결
    for i, pair_idx in enumerate(pair_idxs):
        # legend에 사용할 label을 기본적으로 비워 둠
        label = None
        # 첫 번째 선에만 전체 pair 개수 label을 붙임
        if i == 0:
            # legend에 표시할 pair 개수 문자열을 만듦
            label = '# pairs = {}'.format(len(pair_idxs))
            
        # 현재 pair의 첫 번째 마이크 좌표를 가져옴
        mic_0 = points[pair_idx[0]]
        # 현재 pair의 두 번째 마이크 좌표를 가져옴
        mic_1 = points[pair_idx[1]]
        # 두 마이크를 빨간 선으로 연결해 시각화
        ax.plot([mic_0[0], mic_1[0]], [mic_0[1], mic_1[1]], 'r', label=label)

    # 그래프에 legend를 표시
    ax.legend()
    # 저장 파일명이 전달되었는지 확인
    if filename:
        # 현재 figure를 지정된 파일명으로 저장
        plt.savefig(filename)

    # 사용자가 추가 조작할 수 있도록 axis 객체를 반환
    return ax


def dict_to_float(dict_of_tensors):
    """딕셔너리 안의 모든 Tensor를 float 타입으로 변환하는 함수
    Args:
        dict: A dictionary of tensors.
    """

    # 딕셔너리의 key와 value를 하나씩 순회
    for key, value in dict_of_tensors.items():
        # value가 PyTorch Tensor인지 확인
        if isinstance(value, torch.Tensor):
            # Tensor이면 float32 타입으로 변환
            value = value.float()
        # value가 중첩 딕셔너리인지 확인
        elif isinstance(value, dict):
            # 중첩 딕셔너리라면 재귀적으로 내부 Tensor들을 float으로 변환
            value = dict_to_float(value)
        # 변환된 value를 원래 key 위치에 다시 저장
        dict_of_tensors[key] = value

    # 모든 Tensor가 float으로 변환된 딕셔너리를 반환
    return dict_of_tensors


# 지정한 이름의 폴더가 없으면 생성하는 함수
def create_folder(folder_name):
    # 해당 경로가 이미 존재하는지 확인
    if not os.path.exists(folder_name):
        # 폴더가 없다는 메시지를 출력
        print('{} folder does not exist, creating it.'.format(folder_name))
        # 필요한 중간 폴더까지 포함해 폴더를 생성
        os.makedirs(folder_name)


# 현재 실행 환경에서 사용할 PyTorch device를 선택하는 함수
def get_device(allow_mps=True):
    # 기본 device는 CPU로 설정
    device = "cpu"
    if torch.backends.mps.is_available() and allow_mps:
        device = "mps"
    if torch.cuda.is_available():
        # CUDA가 가능하면 device를 cuda로 설정
        device = "cuda"
    # 문자열 device 이름을 torch.device 객체로 변환해 반환
    return torch.device(device)


# params.json을 읽고 파생 파라미터를 추가한 뒤 전체 파라미터를 반환하는 함수
def get_params():
    # 기본 파라미터를 읽어오는 구간임을 표시하는 주석
    # ########### default parameters ##############

    # 현재 작업 디렉터리의 params.json 파일을 읽어 파이썬 딕셔너리로 변환
    params = json.load(open("params.json", "r"))

    # params 딕셔너리에 필요한 파생 파라미터를 추가하는 구간
    # Parameter manipulation
    # dataset 설정 안의 tau_nigens 관련 파라미터를 가져옴
    tau_params = params["dataset"]["tau_nigens"]
    # label hop 길이가 feature hop 길이의 몇 배인지 정수 비율로 계산
    feature_label_resolution = int(tau_params["label_hop_len_s"] // tau_params["hop_len_s"])
    # label sequence length와 resolution을 곱해 feature sequence length를 계산
    params["feature_sequence_length"] = (
        # label sequence length에 feature-label resolution을 곱
        tau_params["label_sequence_length"] * feature_label_resolution
    )

    # 최상위 파라미터 key와 value를 하나씩 출력
    for key, value in params.items():
        # 현재 key와 value를 보기 좋게 출력
        print("\t{}: {}".format(key, value))
    # 파생값이 추가된 params 딕셔너리를 반환
    return params


# 정n각형의 꼭짓점 좌표를 생성하는 함수
def generate_regular_polygon(n_sides, radius=1):
    """Generate a regular polygon with n_sides sides and radius radius."""

    # 꼭짓점 좌표를 담을 리스트를 초기화
    points = []
    # 0부터 n_sides-1까지 각 꼭짓점을 순회
    for i in range(n_sides):
        # 현재 꼭짓점의 x 좌표를 원 위의 cos 값으로 계산
        x = radius * math.cos(2 * math.pi * i / n_sides)
        # 현재 꼭짓점의 y 좌표를 원 위의 sin 값으로 계산
        y = radius * math.sin(2 * math.pi * i / n_sides)
        # 계산한 x, y 좌표를 꼭짓점 리스트에 추가
        points.append([x, y])

    # 꼭짓점 리스트를 PyTorch Tensor로 변환해 반환
    return torch.Tensor(points)
