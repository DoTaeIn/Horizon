# Horizon

게임 월드 데이터의 무손실 압축·복원을 연구하는 Python 프로토타입입니다. uint16 높이맵(2D `.npy`)에 범위를 좁혀 코덱 구현, 복원 최적화, 실제 지형·디스크 읽기 비교, 256청크 디스크 저장, 임의 ROI·연속 이동 로딩, 겹치는 청크를 재사용하는 애플리케이션 캐시와 전체 맵 상주 비교까지 진행했습니다. 장기 계획은 [PLAN.md](PLAN.md)에 있습니다.

## 현재까지의 결과

- 직접 Zstd, LZ4, 행 차분+Zstd의 압축·복원과 CLI(`pack`/`unpack`/`benchmark`)를 구현했습니다.
- v1 컨테이너의 shape·값 바이트를 정확히 보존하며, CRC-32와 SHA-256으로 손상을 검사합니다.
- 공개 Skadi 한국·알프스 DEM(각 3601×3601)과 중앙 크롭으로 이전·현재 코덱의 복원 성능을 비교했고, 파일 캐시 사용/Windows OS 캐시 우회 디스크 읽기도 측정했습니다.
- **오루알사의 마을·도로 구조를 참고한 독창적 합성 높이맵**(실제 게임 에셋 미사용)으로 전체 대 청크 압축, 256청크 실제 디스크 저장(정렬된 부분 읽기와 임의의 경계 교차 ROI), 연속 이동 120단계×3회 반복×2시드×3코덱×2 I/O 모드를 측정했습니다.
- **겹치는 청크를 재사용하는 제한된 LRU 캐시, 캐시 없는 임의 ROI 읽기, 한 번 복원해 들고 있는 전체 맵 상주**를 같은 기록 경로를 재생해 비교했습니다. 두 시드 합계 51,840회 로딩과 독립 프로세스 216회 메모리 측정을 Codex가 별도로 검산했습니다.
- **현재 macOS 테스트 실행: 381개 중 376개 통과, 5개 환경별 건너뛰기**(Windows 전용 4개, 비-Darwin 전용 1개). 위 복원·디스크 읽기 DEM 결과는 **과거 Windows 환경의 역사적 기록**이며, 그 시점 테스트는 130개 중 127개 통과·3개 건너뛰기였습니다(아래 "환경별 차이" 참고).

개발은 Claude Code(Opus 5·Sonnet 5)의 구현 작업과 별도 감사·실측으로 진행했습니다. 성능 퇴행, 캐시 검증 누락, 과도한 버퍼 할당, 고정된 요청 순서·좌표 분포 편향, 경계를 넘지 않는 이동 경로 등을 찾아 수정했으며, 반려한 결과는 각 감사 보고서에서 최종 측정과 구분합니다.

### 복원 최적화 (2026-10-12, Windows DEM, 역사적 기록)

같은 실행에서 이전 코덱과 현재 코덱을 교대로 50회 측정했습니다. 전체 지형의 차분+Zstd 결과입니다.

| 지형 | 이전 중앙값 | 현재 중앙값 | 시간 감소 | 추적 가능한 피크 메모리 |
| --- | ---: | ---: | ---: | ---: |
| 한국 N37E127 | 98.82ms | 85.55ms | 13.44% | 85.80 → 60.39MB |
| 알프스 N46E007 | 100.24ms | 86.46ms | 13.74% | 86.55 → 61.13MB |

작은 합성 맵은 사실상 이전 수준입니다. 메모리는 `tracemalloc` 측정이며 프로세스 RSS와 추적되지 않는 네이티브 할당은 제외합니다. [원시 결과](benchmarks/decode-real-v2.json) · [감사](benchmarks/decode-audit.md)

### 디스크 읽기+복원 (2026-10-12, Windows DEM, 역사적 기록)

직접 Zstd와 차분+Zstd를 조건·방식별 50회씩, 총 400회 측정한 중앙값입니다. 각 호출은 파일 열기·읽기·닫기와 복원·내부 체크섬을 포함합니다.

| 전체 지형·조건 | 직접 Zstd | 차분+Zstd |
| --- | ---: | ---: |
| 한국·캐시 사용 | 82.84ms | 86.58ms |
| 알프스·캐시 사용 | 84.60ms | 86.96ms |
| 한국·OS 캐시 우회 | 92.72ms | 92.88ms |
| 알프스·OS 캐시 우회 | 93.54ms | 94.71ms |

**차분 방식의 확실한 전체 로딩 속도 우위는 확인하지 못했습니다.** 캐시 사용 시 직접 Zstd가 빨랐고, 캐시 우회 시에는 거의 동률이었습니다. 차분 컨테이너는 직접 Zstd보다 한국 33.50%, 알프스 40.77% 작았습니다. 측정 환경은 단일 Windows 장치, Python 3.10.6, NumPy 2.2.6, zstandard 0.25.0, lz4 4.4.5입니다. OS 캐시 우회는 SSD·컨트롤러 캐시나 메타데이터까지 비운 완전한 콜드 로딩이 아닙니다. [디스크 결과](benchmarks/disk-load-real-v2.json) · [감사 보고서](benchmarks/disk-load-audit.md)

### 합성 마을·도로 맵: 전체 대 청크 압축 (macOS)

실제 게임 맵·메시·텍스처를 쓰지 않은 **독창적인 합성 높이맵**(계단식 평원·절벽, 광장·건물 부지, 하천, 다리·여울, 연결 도로, 시드 기반 세부 기복)입니다. 기본 입력은 1024×1024, seed=20261012. Zstd/LZ4/차분+Zstd × 청크 변 64/128/256/512, 맵 1개당 12개 설정을 기본 맵·다른 시드 1024 맵·1025 경계 맵 세 입력 리포트에 걸쳐 총 36개 설정, 설정별 30회 측정했습니다.

차분+Zstd 단일 전체 컨테이너는 367.03 KiB로 직접 Zstd(486.47 KiB)보다 24.55% 작지만 복원은 직접 Zstd가 더 빠릅니다. 256청크는 인덱스 포함 총 용량이 전체 대비 +0.12%에 그치면서 한 청크 조회·복원이 0.248ms로, 이후 실험의 출발점으로 선택했습니다. 환경: macOS 26.6.2 arm64, Python 3.13.9, NumPy 2.5.4. [결과](benchmarks/town-1024-v1.json) · [감사](benchmarks/town-audit.md)

### 256청크 실제 디스크 저장: 전체 대 부분 로딩 (macOS)

같은 합성 맵 생성기의 **서로 다른 시드 두 개**(20261012, 20261013)를 각각 단일 컨테이너와 256×256 청크(맵당 16개) 컨테이너로 실제 파일에 저장해, 3코덱×2 캐시 조건(`warm_buffered`, `macos_nocache`)×2 작업(전체/부분)×2 저장 방식을 50회씩, 두 맵 합계 2,400회 로딩 표본으로 측정했습니다. 캐시 준비 조건에서 정렬된 단일 청크의 부분 로딩은 전체 로딩 대비 **약 14.7배** 빨랐고, 전체 로딩 자체는 청크 방식이 약 1.3% 느려 이득이 없었습니다. **F_NOCACHE 요청은 캐시 비우기를 검증한 콜드 하드웨어 읽기가 아니며**, HCIX는 여전히 실험용 포맷으로 하이트맵 전용 범위입니다. [결과](benchmarks/town-disk-1024-v1.json) · [감사](benchmarks/town-disk-audit.md)

### 임의 경계 교차 ROI와 연속 이동 120단계 (macOS)

`horizon/town_region.py`의 `load_region()`은 청크 경계와 무관한 임의 사각형 ROI를 겹치는 청크만 읽어 합성합니다. 걷기·경계 왕복·순간이동 세 경로를 120단계×3회 반복×2시드×3코덱×2 I/O 모드로, 총 25,920회 로딩을 측정했습니다.

대표 결과 (시드 20261012, `warm_buffered` 모드, 전체·부분은 요청당 중앙값):

| 코덱 | 이동 | 전체(ms) | 부분(ms) | 속도 비율 | 읽기량 감소 |
|---|---|---:|---:|---:|---:|
| delta_zstd | 걷기 | 4.470 | 1.051 | 4.25× | 81.3% |
| delta_zstd | 경계 왕복 | 4.561 | 1.177 | 3.88× | 74.6% |
| delta_zstd | 순간이동 | 4.750 | 1.115 | 4.26× | 81.4% |

경계를 넘어 여러 청크를 읽는 경계 왕복은 이전의 정렬된 단일 청크 조회(약 14.7배)보다 이득이 작습니다. **이 단계까지는 애플리케이션 캐시를 두지 않았습니다**(캐시 비교는 아래 캐시 섹션에서 진행했습니다). [결과](benchmarks/town-movement-1024-v1.json) · [감사](benchmarks/town-movement-audit.md)

### 제한된 청크 캐시 대 전체 맵 상주 (2026-10-13, macOS)

`horizon/town_cache.py`의 `ChunkRegionCache`(용량 제한 LRU로 복원된 청크 재사용)와 `WholeMapResident`(시작할 때 한 번 전체를 읽어 복원해 보관)를, 캐시 없는 `load_region()`과 같은 기록 경로 위에서 비교했습니다. `scripts/compare_town_cache.py`는 새 맵이나 새 경로를 만들지 않고 기존 이동 리포트의 좌표를 그대로 재생합니다. 두 시드(20261012·20261013), 3코덱, 2 I/O 모드, 세 경로를 256청크·256×256 시야·120단계×3회 반복으로 돌려 **총 51,840회 로딩**을, 메모리는 매 표본을 새 프로세스에서 측정한 **216회 독립 측정**으로 분리했습니다.

시드 20261012, delta+Zstd, `warm_buffered`, 요청당 **중앙값 / p95 ms**(360개 요청):

| 방식 | 걷기 | 경계 왕복 | 순간이동 | 보관 배열 상한 |
|---|---:|---:|---:|---:|
| stateless | 0.9010 / 1.0263 | 0.9749 / 1.0761 | 0.9243 / 1.0690 | 0.00 MiB |
| lru4 | 0.0080 / 0.2991 | 0.4726 / 0.5334 | 0.4958 / 0.9865 | 0.50 MiB |
| lru8 | 0.0077 / 0.2663 | 0.0082 / 0.0136 | 0.2645 / 0.8500 | 1.00 MiB |
| whole_resident | 0.0037 / 0.0097 | 0.0033 / 0.0079 | 0.0075 / 0.0117 | 2.00 MiB |

준비 포함 120단계 로더 시간 합계의 중앙값(ms, 반복별 합계를 구한 뒤 중앙값):

| 이동 | stateless | lru4 | lru8 | whole_resident |
|---|---:|---:|---:|---:|
| 걷기 | 87.431 | 4.866 | 4.576 | 4.667 |
| 경계 왕복 | 118.401 | 58.397 | 2.582 | 4.664 |
| 순간이동 | 89.385 | 68.246 | 41.418 | 4.992 |

**이번 측정에서 전체 맵 상주의 요청당 중앙값이 가장 짧았고, 순간이동의 초기화 포함 합계도 가장 작았습니다.** 8청크 캐시는 경계 왕복에서 적중률 98.75%로 합계 2.582ms까지 내려가 이 경로에서는 전체 상주(4.664ms)보다 좋았고, 걷기는 4.576ms 대 4.667ms로 사실상 비슷합니다. 반면 순간이동은 41.418ms 대 4.992ms로 크게 뒤집힙니다. 첫 시야까지는 LRU가 약 0.51~1.03ms, 전체 상주가 약 4.08~4.21ms입니다. **어느 쪽도 보편적 우위가 아닙니다**: 작업 경로와 허용 메모리에 따라 결과가 바뀝니다.

4청크 캐시의 순간이동 최대 요청은 31.677208ms였습니다(같은 셀 p95는 0.9865ms). 두 번째 시드에서도 8청크 순간이동 최대가 29.5486ms로 같은 성격의 단발 이상치가 나타났습니다. **원인은 귀속하지 않았고 표본도 제거하지 않았습니다.** 낮은 중앙값을 프레임 지연 보장으로 읽지 않습니다. 두 번째 시드의 방향은 동일했습니다(8청크 요청 중앙값 걷기 0.0079·경계 왕복 0.0110·순간이동 0.4639ms, 전체 상주 0.0040·0.0073·0.0100ms).

메모리는 시간과 분리해 측정했습니다. 시드 20261012, delta+Zstd, 걷기, 독립 프로세스 3회 중앙값(MiB):

| 방식 | 보관 복원 배열 피크 | tracemalloc 준비 피크 | tracemalloc 이동 피크 | RSS HWM 기준 | RSS HWM 전체 |
|---|---:|---:|---:|---:|---:|
| stateless | 0.000 | 0.000 | 0.629 | 35.781 | 38.375 |
| lru4 | 0.500 | 0.007 | 0.942 | 36.234 | 37.469 |
| lru8 | 1.000 | 0.007 | 1.443 | 36.062 | 38.516 |
| whole_resident | 2.000 | 5.578 | 2.135 | 35.750 | 40.875 |

보관 배열 크기는 추정이 아니라 정확한 값입니다. 4청크 0.5MiB·8청크 1.0MiB가 상한이고 전체 상주는 2.0MiB이며, 경계 왕복의 8청크는 필요한 청크가 여섯 개여서 실제 보관 피크가 0.75MiB였습니다. 호출자에게 돌려주는 256×256 출력 배열 0.125MiB는 세 방식 모두 동일하므로 이 열에서 제외했습니다. **tracemalloc은 추적되지 않는 네이티브 코덱 할당을 제외**하며, **RSS는 Python 인터프리터·NumPy·코덱 import를 포함한 프로세스 전체 high-water mark이고 "기준" 값도 현재 RSS가 아니라 이미 HWM**입니다. 그래서 두 수치는 따로 적었고, **캐시가 항상 RSS를 줄인다고 주장하지 않습니다**.

전체 테스트 381개 중 376개 통과·5개 환경별 건너뛰기. [최종 결과](benchmarks/town-cache-1024-v2.json) · [다른 시드](benchmarks/town-cache-1024-seed2-v2.json) · [메모리 사이드카](benchmarks/town-cache-1024-v2.memory.json) · [최종 감사](benchmarks/town-cache-final-audit.md) · [독립 검산 증거](benchmarks/town-cache-final-evidence.json) · [측정 계약](docs/town-cache-method.md)

### 다음 단계

실제 수제 게임 높이맵, 검증된 콜드 부팅, 다른 저장장치·하드웨어, 엔진·GPU·충돌 준비까지의 로딩, 사전 Zstd, 규칙 추론과 월드 생성은 아직 검증하거나 구현하지 않았습니다. 캐시 단계에서 새로 남은 질문은 메모리보다 큰 맵과 여러 맵 동시 상주, 비동기 미리 가져오기, 캐시 교체 정책 변형, 프레임 지연 상한 보장입니다. 현재 결과를 게임 전체 로딩 성능으로 일반화하지 않습니다.

## 파일 구성

- `horizon/benchmark.py`: `run_benchmark(size=256, repeats=30, inputs=())`
- `horizon/__main__.py`: `python -m horizon` CLI (`pack`/`unpack`/`benchmark`)
- `horizon/codec.py`: `encode(heightmap, method="delta_zstd")`, `decode(blob)`
- `horizon/town_map.py`: 독창적 합성 마을·도로 높이맵 생성기
- `horizon/chunk_benchmark.py`: 전체 대 청크 압축/복원 비교 로직
- `horizon/town_disk.py`: 단일/256청크 실제 디스크 컨테이너 쓰기·읽기(HCIX)
- `horizon/town_region.py`: 임의 경계 교차 ROI 로더 `load_region(path, region, mode)`
- `horizon/town_cache.py`: 용량 제한 LRU 청크 캐시 `ChunkRegionCache(path, max_chunks, mode)`와 전체 맵 상주 `WholeMapResident(path, mode)`
- `scripts/`: 공개 DEM 준비(`prepare_terrain.py`), 복원 비교(`compare_decode.py`), 디스크 읽기 비교(`compare_disk.py`), 합성 맵 전체/청크 비교(`compare_town.py`), 256청크 디스크 비교(`compare_town_disk.py`), 연속 이동 비교(`compare_town_movement.py`), 캐시 대 전체 상주 비교(`compare_town_cache.py`)와 그 메모리 측정 워커(`measure_town_cache_memory.py`)
- `tests/`: 코덱 정확도, 손상 입력, 벤치마크, CLI, 청크 디스크, ROI·이동, 캐시 모듈·캐시 CLI 테스트
- `benchmarks/`: 합성·실제 지형 측정 결과와 감사 보고서 (자세한 내용은 [benchmarks/README.md](benchmarks/README.md))
- `docs/`: 합성 맵·청크 디스크·이동 측정의 실행 계약과 포맷 설명

### Linux·macOS·Windows 차이

- 코덱(`horizon/codec.py`), 합성 맵 생성, 청크 압축 비교, ROI 로더는 Python/NumPy 기반으로 세 플랫폼에서 동일하게 동작합니다.
- `scripts/compare_disk.py`의 Windows OS 캐시 우회(`FILE_FLAG_NO_BUFFERING`)는 **Windows 전용**이며, 실패 시 전체 측정이 실패하고 일반 읽기로 대체하지 않습니다.
- `scripts/compare_town_disk.py`·`scripts/compare_town_movement.py`·`scripts/compare_town_cache.py`의 `macos_nocache` 모드는 `fcntl(fd, F_NOCACHE, 1)`을 사용하는 **macOS(Darwin) 전용**입니다. 이 스크립트들은 Windows·Linux에서 `warm_buffered` 모드만 실행하며, `os_cache_bypass`는 지원하지 않습니다(그 모드는 `compare_disk.py` 전용입니다).
- `compare_town_cache.py`는 기록된 모드 목록에서 지원하지 않는 모드만 걸러내지만, `--modes`로 **명시한** 모드가 이 플랫폼에서 불가능하면 다른 모드로 조용히 바꾸지 않고 거부합니다.
- 현재 macOS 테스트 381개 중 건너뛰는 5개는 Windows 전용 4개와 비-Darwin 전용 1개입니다. 과거 Windows 테스트(130개 중 127개 통과·3개 건너뛰기)는 Windows 캐시 우회 검사를 실제 실행하고, 심볼릭 링크 권한 검사 2개와 비-Windows 전용 검사 1개를 건너뛴 결과였습니다.

## 설치와 테스트

```
uv sync
uv run python -m unittest discover -s tests -v
```

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다. 의존성 잠금은 `uv.lock`에 포함됩니다.

공개 저장소에는 코드·테스트·측정 JSON·감사 문서·작은 지형 출처 메타데이터를 포함합니다. 대용량 지형 원본, 생성 컨테이너, 가상환경, Claude 실행 로그는 제외합니다.

실제 DEM 데이터는 새 디렉터리로 받습니다. 저장소에 있는 `data/terrain/*.json`은 당시 측정의 출처·해시 기록이며, 해당 경로에는 원본 파일을 배포하지 않습니다.

```powershell
uv run python scripts/prepare_terrain.py --output-dir data/terrain-local
uv run python scripts/compare_decode.py --input data/terrain-local/N37E127.npy --input data/terrain-local/N46E007.npy --repeats 50 --output benchmarks/my-decode.json
```

데이터 출처·저작자 표시·무손실 정규화·재현 방법은 [지형 문서](data/terrain/README.md)를 참고하세요.

## 사용법

벤치마크 (출력 파일의 상위 폴더가 미리 존재해야 합니다):

```
uv run python -m horizon benchmark --output benchmarks/example.json
uv run python -m horizon benchmark --size 256 --repeats 30
uv run python -m horizon benchmark --input real_map.npy --input another.npy
```

`--output`을 생략하면 JSON 리포트를 표준 출력으로 내보냅니다.

압축/복원:

```
uv run python -m horizon pack INPUT.npy OUTPUT.hzn --method delta_zstd
uv run python -m horizon unpack INPUT.hzn OUTPUT.npy
```

사용 가능한 방식: `zstd`, `lz4`, `delta_zstd`(기본값).

### 디스크 읽기 포함 비교 (Windows DEM)

`warm_buffered`는 매번 측정 직전에 파일 전체를 읽어 캐시를 준비하고, `os_cache_bypass`는 Windows `FILE_FLAG_NO_BUFFERING`으로 OS 파일 데이터 캐시를 우회합니다. SSD·컨트롤러 캐시나 파일 메타데이터까지 비운 콜드 부팅 측정은 아닙니다.

```powershell
uv run python scripts/compare_disk.py --input data/terrain-local/N37E127.npy --input data/terrain-local/N46E007.npy --containers-dir data/disk-load-new --repeats 50 --output benchmarks/disk-load-new.json
```

출력 JSON과 컨테이너 디렉터리는 모두 새 경로여야 합니다. 큰 컨테이너 파일은 `data/disk-load-*/`에서 Git 제외됩니다.

### 합성 마을·도로 맵: 전체 대 청크 압축

```sh
uv run python scripts/compare_town.py --size 1024 --seed 20261012 --repeats 30 --output benchmarks/my-town.json --preview-dir data/terrain-local/my-town
uv run python scripts/compare_town.py --size 1024 --seed 20261013 --repeats 30 --output benchmarks/my-town-seed2.json
uv run python scripts/compare_town.py --size 1025 --seed 20261012 --repeats 30 --output benchmarks/my-town-edge.json
```

`--output`은 필수이며 새 경로여야 합니다. `--preview-dir`은 선택이며 지정하면 SVG 안내 이미지와 `.npy` 덤프를 생성합니다(기존 파일은 덮어쓰지 않음). `--chunk-sizes`로 비교할 청크 변 목록을 바꿀 수 있습니다(기본 64/128/256/512).

### 256청크 실제 디스크 저장: 전체 대 부분 로딩

```sh
uv run python scripts/compare_town_disk.py --size 1024 --seed 20261012 --repeats 50 --output benchmarks/my-town-disk.json --containers-dir data/terrain-local/my-town-disk
uv run python scripts/compare_town_disk.py --size 1024 --seed 20261013 --repeats 50 --output benchmarks/my-town-disk-seed2.json --containers-dir data/terrain-local/my-town-disk-seed2
```

`--output`과 `--containers-dir`은 모두 필수이며 새 경로여야 합니다. `--modes`로 측정할 캐시 조건을 고를 수 있습니다. 이 스크립트가 지원하는 모드는 `warm_buffered`와 macOS 전용 `macos_nocache`뿐입니다(기본값은 macOS에서 둘 다, 그 외 플랫폼은 `warm_buffered`만). `os_cache_bypass`(Windows `FILE_FLAG_NO_BUFFERING`)는 이 스크립트에는 없으며, 위의 구형 `scripts/compare_disk.py`에만 존재합니다.

### 임의 ROI·연속 이동 로딩

```sh
uv run python scripts/compare_town_movement.py --size 1024 --seed 20261012 --steps 120 --repeats 3 --output benchmarks/my-town-movement.json --containers-dir data/terrain-local/my-town-movement
```

`--output`과 `--containers-dir`은 모두 필수이며 새 경로여야 합니다. `--steps`는 경로당 이동 단계 수(기본 120), `--repeats`는 경로별 반복 횟수(기본 3), `--modes`는 디스크 비교와 동일한 캐시 조건 목록입니다.

### 제한된 캐시 대 전체 맵 상주

```sh
uv run python scripts/compare_town_cache.py --schedule benchmarks/town-movement-1024-v1.json --repeats 3 --cache-sizes 4,8 --memory-repeats 3 --output benchmarks/my-town-cache.json --containers-dir data/terrain-local/my-town-cache
uv run python scripts/compare_town_cache.py --schedule benchmarks/town-movement-1024-seed2-v1.json --repeats 3 --cache-sizes 4,8 --memory-repeats 3 --output benchmarks/my-town-cache-seed2.json --containers-dir data/terrain-local/my-town-cache-seed2
```

`--schedule`은 필수이며 기존 `compare_town_movement.py` 리포트여야 합니다. 이 스크립트는 맵도 경로도 새로 만들지 않고, 기록된 `(size, seed)`로 맵을 다시 만들어 해시를 대조한 뒤 기록된 좌표를 그대로 재생합니다. 기록된 `source_files_sha256`의 **모든** 파일(그 리포트를 만든 `compare_town_movement.py` 포함)이 그대로 있고 바이트까지 동일해야 하며, 없거나 달라지면 실행을 거부합니다. 청크 변·시야는 기록된 값을 쓰고(위 측정은 256·256), 기본값은 `--repeats 3`, `--cache-sizes 4,8`, `--memory-repeats 3`입니다. `--memory-repeats 0`은 메모리 하위 측정을 끄고, 그 외 값은 `scripts/measure_town_cache_memory.py`를 하위 프로세스로 호출해 메모리 사이드카를 만듭니다. 사이드카 경로는 `--output`의 확장자를 바꾼 것이어서 `my-town-cache.json`의 사이드카는 `my-town-cache.memory.json`입니다(이름 뒤에 덧붙이지 않습니다). `--output`·사이드카·`--containers-dir` 모두 기존 경로를 덮어쓰지 않습니다. 위 최종 측정은 `benchmarks/town-cache-1024-v2.json`·`benchmarks/town-cache-1024-seed2-v2.json`과 `data/terrain-local/town-cache-1024-v2/`·`data/terrain-local/town-cache-1024-seed2-v2/`에 기록했으므로, 재실행할 때는 반드시 다른 출력·컨테이너 경로를 지정하세요.

## 알아둘 점

- 입력은 비어 있지 않은 2D little-endian **uint16** 배열만 허용합니다. 암묵적 dtype 변환은 없습니다. 원본 크기 상한은 64 MiB입니다.
- 값 바이트와 shape는 그대로 보존되지만 `.npy` 헤더 바이트까지 동일함을 보장하지는 않습니다.
- 컨테이너 포맷 버전은 고정되어 있으며, 손상/잘림/뒤따르는 데이터는 `ValueError`로 거부됩니다.
- 포맷 v1은 60바이트 헤더에 크기·코덱·예측기·원본 SHA-256·헤더 CRC-32를 저장합니다. `delta_zstd`는 행 첫 값을 유지하고 나머지 값을 왼쪽 이웃과의 차분(mod 65536)으로 저장한 뒤 Zstd로 압축합니다.
- 기존 출력 파일은 절대 덮어쓰지 않습니다. 대상이 이미 있으면 오류와 함께 0이 아닌 종료 코드로 끝납니다.
- 합성 마을·도로 높이맵은 **독창적인 생성기 출력**이며 실제 게임 맵·ROM·메시·텍스처를 사용하지 않습니다. 건물 부지는 건물 메시가 아니고, 교량 아래·위 공간을 동시에 표현하지 못합니다.
- 256청크 디스크 포맷(HCIX)은 **실험용**입니다. 전체 읽기(`town_disk`)와 전체 인덱스 로딩(`town_region`)은 헤더와 인덱스가 격자·범위·연속성·커버리지를 만족하는지 구조적으로 검증하고, 각 청크는 자신의 코덱 헤더와 인덱스 항목을 상호 대조합니다. 다만 **인덱스 자체에는 체크섬이 없습니다**. 실제로 읽은 청크 페이로드는 기존 코덱 체크섬으로 검증되지만, 읽지 않은 다른 청크 페이로드의 손상은 검출하지 못합니다. 범위는 하이트맵 전용이며 엔진·GPU·충돌 준비는 포함하지 않습니다.
- `macos_nocache`의 `fcntl(F_NOCACHE, 1)`과 Windows `FILE_FLAG_NO_BUFFERING`은 **애플리케이션 캐시가 아니라 OS 데이터 캐싱 동작에 대한 요청**입니다. 이전 디스크·이동 비교에는 애플리케이션 자체 캐시가 없었고, 애플리케이션 캐시를 명시적으로 비교하는 것은 `compare_town_cache.py`의 `lru{N}`·`whole_resident` 비교군뿐입니다. 두 요청 모두 **캐시를 비웠다고 검증한 콜드 하드웨어 읽기가 아니며**, SSD·컨트롤러 캐시나 메타데이터까지 비우지 않습니다.
- 연속 이동 로딩(`compare_town_movement.py`)은 **애플리케이션 캐시나 미리 가져오기를 두지 않은** 로더 비교입니다. 애플리케이션 캐시는 그다음 단계인 `compare_town_cache.py`에서만 비교하며, 두 단계 모두 렌더러·물리·비동기 스케줄링을 포함한 게임 프레임 측정이 아닙니다.
- 캐시 비교의 120단계 합계는 **로더 시간의 합**이며 실제 120프레임 벽시계 시간이 아닙니다. 전체 상주의 요청당 비용이 0에 가까운 것은 이 작은 2MiB 맵 하나를 프로세스 내내 들고 있기 때문이고, 메모리보다 큰 맵이나 여러 맵 동시 상주는 검증하지 않았습니다.
- 캐시 메모리는 세 가지를 구분해 적습니다. **보관 복원 배열 크기는 정확한 값**, **tracemalloc은 추적되지 않는 네이티브 할당을 제외한 Python 수준 값**, **RSS는 인터프리터·import까지 포함한 프로세스 전체 high-water mark**(기준값도 현재 RSS가 아니라 HWM)입니다. 세 값의 차이를 캐시의 순 메모리로 해석하지 않으며, 캐시가 항상 RSS를 줄인다고 주장하지 않습니다.
- 측정값은 맵 전체 또는 ROI 컨테이너의 로딩/압축/해제 시간이며, 스트리밍이나 엔진 성능을 의미하지 않습니다.

## 리포트 형식

`schema_version`, `environment`(Python, 플랫폼, NumPy/zstandard/lz4 버전), `settings`(size, repeats, warmup, seed, methods), `results`로 구성됩니다. `results`의 각 항목은 맵/방식별로 `sha256`, `shape`, `raw_bytes`, `container_bytes`, `ratio`(container/raw), `encode_median_ms`, `decode_median_ms`, `decode_p95_ms`, `exact_match`를 가집니다. 모든 반복의 디코드 결과는 시간 측정 구간 밖에서 바이트와 shape가 정확히 일치하는지 검사합니다.

청크·디스크·이동·캐시 측정의 리포트 형식과 실행 계약은 [docs/town-benchmark-method.md](docs/town-benchmark-method.md), [docs/town-disk-method.md](docs/town-disk-method.md), [docs/town-movement-method.md](docs/town-movement-method.md), [docs/town-cache-method.md](docs/town-cache-method.md)에 각각 설명되어 있습니다.

캐시 리포트(`town-cache-*-v2.json`)는 크기가 커서 공백 없는 compact JSON으로 기록합니다. 프레임·반복 단위 원시 값은 정수 나노초(`*_ns`)이고, 집계는 모두 밀리초 통계(`median_ms`/`p95_ms`/`max_ms`)이므로 집계 이름에는 `_ns`를 붙이지 않습니다.

모든 측정 결과와 감사 보고서는 [benchmarks/README.md](benchmarks/README.md)에 정리했습니다.
