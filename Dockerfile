# Python 3.12 slim 기반
FROM python:3.12-slim

# 필수 패키지 (빌드/런타임 유틸)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사/설치 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

# 데이터/임베딩 영속화 경로
VOLUME ["/data"]

# 컨테이너 포트 (정보용)
EXPOSE 5000

# uvicorn 실행
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]