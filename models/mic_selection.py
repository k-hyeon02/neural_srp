# 기하학적으로 마이크 쌍을 선택하는 기능
# M개의 마이크 위치 목록이 주어지면,
# 가능한 (M choose 2)개의 쌍 중 일부를 선택
# 선택은 각 마이크 쌍의 각도를 기준으로
# 각도는 두 마이크를 잇는 직선과
# x축 사이의 각도로 정의
# 각 쌍 벡터의 코사인 유사도를 비교하여 선택
# 길이가 가장 큰 쌍만 유지

import torch 

EPS = 1e-6  # 수치 안정성을 위한 작은 상수 정의


# 마이크 위치에서 사용할 쌍 인덱스 선택
def select_pairs(mic_positions, mode="distinct_angles", threshold=0.05):
    device = mic_positions.device  # 입력 텐서의 원래 디바이스 저장
    mic_positions = mic_positions.to("cpu").detach()  # CPU 이동 및 연산 그래프 분리

    if mode not in [  # 허용된 선택 모드 여부 확인
        "distinct_angles",  # 각도 중복 제거 모드명
        "random",  # 무작위 선택 모드명
        "first",  # 앞쪽 쌍 선택 모드명
        "all"  # 전체 쌍 선택 모드명
    ]:
        raise NotImplementedError(  # 미구현 모드 입력 시 예외 발생
            f"""Mode {mode} not implemented.
                Choose from 'distinct_angles' 'random', 'first' or 'all'."""
        )

    # 가능한 모든 쌍 생성
    M = len(mic_positions)  # 마이크 개수 계산
    pair_idxs = []  # 마이크 쌍 인덱스 목록 초기화
    for i in range(M):  # 첫 번째 마이크 인덱스 순회
        for j in range(i + 1, M):  # 두 번째 마이크 인덱스 순회
            pair_idxs.append((i, j))  # 중복 없는 마이크 쌍 추가

    pair_idxs = torch.Tensor(pair_idxs).long().to(device)  # 쌍 인덱스 목록의 long 텐서 변환
    P = len(pair_idxs)  # 전체 마이크 쌍 개수 계산

    if mode == "all":  # 전체 쌍 반환 모드 여부 확인
        return pair_idxs  # 모든 마이크 쌍 인덱스 반환
    elif mode == "first":  # 앞쪽 쌍 반환 모드 여부 확인
        return pair_idxs[:M]  # 처음 M개의 마이크 쌍 인덱스 반환

    # 그 외 모드는 각도 기반 대표 쌍 계산
    # 각 쌍의 방향 벡터 계산
    pair_vectors = []  # 마이크 쌍 방향 벡터 목록 초기화
    for pair in pair_idxs:  # 각 마이크 쌍 순회
        pair_vector = mic_positions[pair[1]] - mic_positions[pair[0]]  # 두 마이크 위치 차이 계산 -> 벡터 생성
        pair_vectors.append(pair_vector)  # 방향 벡터 목록에 추가

    pair_vectors = torch.stack(pair_vectors)  # 방향 벡터 목록의 텐서 결합
    
    # 쌍의 쌍마다 각도 거리 계산
    pair_idxs = tuple(tuple(sub) for sub in pair_idxs.tolist())  # 집합 연산용 튜플 구조 변환 -> set 활용을 위함
    out_pair_idxs = set(pair_idxs)  # 출력 후보 쌍 집합 초기화
    pairs_to_discard = set()  # 폐기할 쌍 집합 초기화
    for i in range(P):  # 기준 쌍 인덱스 순회
        if pair_idxs[i] in pairs_to_discard:  # 이미 폐기 예정인 기준 쌍 확인
            continue  # 폐기 예정 기준 쌍 건너뜀
        same_angle_pairs = []  # 같은 각도 쌍 목록 초기화
        lengths_same_angle_pairs = []  # 같은 각도 쌍 길이 목록 초기화
        for j in range(i + 1, P):  # 비교 대상 쌍 인덱스 순회
            if pair_idxs[j] in pairs_to_discard:  # 이미 폐기 예정인 비교 쌍 확인
                continue  # 폐기 예정 비교 쌍 건너뜀
            # 1. 두 쌍의 길이 계산
            length_i = torch.linalg.norm(pair_vectors[i])  # 기준 쌍 벡터 길이 계산
            length_j = torch.linalg.norm(pair_vectors[j])  # 비교 쌍 벡터 길이 계산

            # 2. 두 쌍의 각도 거리 계산
            # 코사인 유사도 사용
            angle_dist = torch.dot(pair_vectors[i], pair_vectors[j]) / (length_i * length_j)  # 코사인 유사도 계산
            angle_dist = torch.abs(angle_dist)  # 방향 부호를 무시한 각도 유사도 계산

            if angle_dist >= 1 - threshold:  # 임계값 안에서 같은 각도인지 판정

                same_angle_pairs.append(pair_idxs[j])  # 같은 각도 비교 쌍 추가
                lengths_same_angle_pairs.append(length_j)  # 같은 각도 비교 쌍 길이 추가

        same_angle_pairs.append(pair_idxs[i])  # 기준 쌍을 같은 각도 목록에 추가
        lengths_same_angle_pairs.append(length_i)  # 기준 쌍 길이를 같은 각도 길이 목록에 추가

        # 같은 각도 쌍을 길이 기준으로 정렬
        same_angle_pairs = torch.Tensor(same_angle_pairs)  # 같은 각도 쌍 목록의 텐서 변환
        lengths_same_angle_pairs = torch.Tensor(lengths_same_angle_pairs)  # 같은 각도 쌍 길이 목록의 텐서 변환
        same_angle_pairs_sorted_idx = torch.argsort(lengths_same_angle_pairs)  # 길이 기준 오름차순 정렬 인덱스 계산
        same_angle_pairs_sorted = same_angle_pairs[same_angle_pairs_sorted_idx]  # 길이 기준 정렬된 쌍 텐서 생성

        # 길이가 가장 큰 하나를 제외한 모든 쌍 폐기
        for pair in same_angle_pairs_sorted[:-1]:  # 최장 쌍을 제외한 같은 각도 쌍 순회
            pair = tuple(pair.tolist())  # 텐서 쌍을 집합 저장 가능한 튜플로 변환
            pairs_to_discard.add(pair)  # 폐기할 쌍 집합에 추가

    # 출력 쌍에서 폐기할 쌍 제거 -> 가장 길이가 긴 쌍만 남음
    # 마이크 배열에서 두 마이크 사이 거리가 길수록 같은 음원 방향에 대해 두 마이크에 도달하는 시간 차이(TDOA)가 더 크게 나타남
    out_pair_idxs = out_pair_idxs - pairs_to_discard  # 최종 출력 쌍 집합 계산

    P_out = len(out_pair_idxs)  # 최종 선택된 쌍 개수 계산
    # print(f"Selected {P_out} pairs out of {P} possible pairs.")  # 선택 결과 디버깅 출력

    if mode == "random":  # 무작위 선택 모드 여부 확인
        pair_idxs = torch.Tensor(pair_idxs).long().to(device)  # 쌍 인덱스 튜플의 long 텐서 복원
        # 공정한 비교를 위해 unique angles 모드와 같은 개수 선택
        # 동일 개수 비교 조건 유지
        random_idxs = torch.randperm(P)  # 전체 쌍 인덱스의 무작위 순열 생성
        return pair_idxs[random_idxs[:P_out]]  # 선택 개수만큼 무작위 쌍 반환

    return torch.Tensor(list(out_pair_idxs)).long().to(device)  # 각도별 최장 쌍 인덱스 반환
