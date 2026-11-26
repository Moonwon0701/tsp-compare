"""TSPLIB 인스턴스 로더와 알려진 최적값 표.

파일 출처: TSPLIB95 (https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
최적값 출처: 같은 사이트의 tsp/TSP-BEST.html
최적 경로 파일(<name>.opt.tour)도 같은 사이트에서 받았다. 테스트에서 이 경로의 길이가
최적값과 정확히 같은지 확인해, 반올림 규칙과 최적값 표가 맞는지 검증한다.

공식 최적값은 TSPLIB EUC_2D 거리(간선마다 반올림한 정수)로 계산된 값이다.
초과율을 낼 때는 경로 길이도 tsp.common.tsplib_distance로 계산해야 한다.
"""

from dataclasses import dataclass
from pathlib import Path

from .common import Point

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "tsplib"

OPTIMA = {
    "eil51": 426,
    "berlin52": 7542,
    "kroA100": 21282,
    "ch150": 6528,
    "a280": 2579,
}
INSTANCES = tuple(OPTIMA)


@dataclass(frozen=True)
class TsplibInstance:
    name: str
    points: list[Point]
    optimum: int | None  # OPTIMA 표에 없으면 None
    comment: str = ""


def parse_tsplib(text: str) -> tuple[dict[str, str], list[Point]]:
    """TSPLIB .tsp 텍스트에서 헤더(키 → 값)와 좌표 목록을 읽는다.

    EUC_2D 형식의 NODE_COORD_SECTION만 지원한다. 노드 번호는 1부터 차례로
    매겨져 있어야 하며, 좌표 목록의 i번째 점이 노드 i+1이다.
    """
    header: dict[str, str] = {}
    points: list[Point] = []
    in_coords = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if in_coords:
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"좌표 줄 형식이 아닙니다: {line!r}")
            node, x, y = int(parts[0]), float(parts[1]), float(parts[2])
            if node != len(points) + 1:
                raise ValueError(f"노드 번호가 차례대로가 아닙니다: {node} (기대 {len(points) + 1})")
            points.append((x, y))
        elif line == "NODE_COORD_SECTION":
            in_coords = True
        elif ":" in line:
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
        else:
            raise ValueError(f"알 수 없는 줄입니다: {line!r}")

    weight_type = header.get("EDGE_WEIGHT_TYPE")
    if weight_type != "EUC_2D":
        raise ValueError(f"EUC_2D만 지원합니다 (EDGE_WEIGHT_TYPE: {weight_type})")
    if "DIMENSION" in header and int(header["DIMENSION"]) != len(points):
        raise ValueError(f"DIMENSION {header['DIMENSION']}과 좌표 개수 {len(points)}가 다릅니다")
    return header, points


def parse_tour(text: str) -> list[int]:
    """TSPLIB .opt.tour 텍스트에서 방문 순서를 읽어 0부터 시작하는 번호로 돌려준다.

    TOUR_SECTION 다음의 노드 번호(1부터)를 -1 또는 EOF가 나올 때까지 읽는다.
    한 줄에 번호가 여러 개 있어도 된다.
    """
    tour: list[int] = []
    in_tour = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "EOF":
            break
        if not in_tour:
            if line == "TOUR_SECTION":
                in_tour = True
            continue
        for token in line.split():
            node = int(token)
            if node == -1:
                return tour
            tour.append(node - 1)
    if not in_tour:
        raise ValueError("TOUR_SECTION이 없습니다")
    return tour


def load_optimal_tour(name: str, data_dir: Path = DATA_DIR) -> list[int]:
    """data/tsplib/<name>.opt.tour의 최적 경로(0부터 시작하는 번호)."""
    path = Path(data_dir) / f"{name}.opt.tour"
    if not path.exists():
        raise FileNotFoundError(f"최적 경로 파일이 없습니다: {path}")
    return parse_tour(path.read_text(encoding="ascii"))


def load_tsplib(name: str, data_dir: Path = DATA_DIR) -> TsplibInstance:
    """data/tsplib/<name>.tsp를 읽어 좌표와 알려진 최적값을 돌려준다."""
    path = Path(data_dir) / f"{name}.tsp"
    if not path.exists():
        raise FileNotFoundError(f"TSPLIB 파일이 없습니다: {path}")
    header, points = parse_tsplib(path.read_text(encoding="ascii"))
    file_name = header.get("NAME", name)
    if file_name != name:
        raise ValueError(f"파일 이름 {name}과 NAME {file_name}이 다릅니다")
    return TsplibInstance(name=name, points=points, optimum=OPTIMA.get(name),
                          comment=header.get("COMMENT", ""))
