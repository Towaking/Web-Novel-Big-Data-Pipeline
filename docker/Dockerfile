# Base image tunggal untuk seluruh compute layer.
#
# Alasan desain:
#  - PySpark wheel sudah membawa distribusi Spark lengkap
#    (bin/, sbin/, jars/), jadi satu image bisa berperan sebagai
#    Spark Master, Worker, Driver (pipeline), sekaligus Streamlit.
#  - Kesamaan image menjamin versi Python & PySpark identik di
#    semua node -> menghindari serialization mismatch saat UDF.
FROM python:3.11-slim-bookworm

ARG HADOOP_AWS_VERSION=3.3.4
ARG AWS_SDK_VERSION=1.12.262

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 \
    SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark \
    PYTHONPATH=/opt/project

# Layer 1: system runtime (JVM untuk Spark)
RUN apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        curl \
        procps \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: PySpark (jarang berubah, dipisah agar cache-nya awet)
# Dipasang lebih dulu dan terpisah dari requirements lain supaya perubahan
# dependency ML (yang sering) tidak memaksa unduh ulang jar 200 MB di bawah.
RUN pip install --upgrade pip && pip install pyspark==3.5.3

# Layer 3: konektor S3A (MinIO)
# Di-bake ke dalam image, BUKAN via --packages saat runtime,
# supaya container tidak bergantung pada koneksi Maven dan
# startup job tetap deterministik/offline-friendly.
#
# aws-java-sdk-bundle berukuran ~200 MB dan sering terputus di tengah
# (curl exit 18). Karena itu unduhan dibungkus retry + resume (-C -) dan
# divalidasi sebagai arsip ZIP yang utuh - file jar yang terpotong lolos
# pengecekan ukuran tapi bikin Spark gagal dengan ClassNotFoundException
# yang menyesatkan saat runtime.
RUN set -eux; \
    fetch_jar() { \
        url="$1"; out="$2"; \
        for attempt in 1 2 3 4 5; do \
            curl -fL --retry 5 --retry-delay 5 --retry-all-errors \
                 --connect-timeout 30 --max-time 1800 -C - -o "$out" "$url" || true; \
            if python -c "import sys, zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)" "$out"; then \
                echo "[jar] OK: $out"; return 0; \
            fi; \
            echo "[jar] percobaan $attempt gagal/terpotong: $out - mengulang"; \
            sleep 5; \
        done; \
        echo "[jar] GAGAL permanen: $url" >&2; return 1; \
    }; \
    fetch_jar "https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_AWS_VERSION}/hadoop-aws-${HADOOP_AWS_VERSION}.jar" \
              "${SPARK_HOME}/jars/hadoop-aws-${HADOOP_AWS_VERSION}.jar"; \
    fetch_jar "https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar" \
              "${SPARK_HOME}/jars/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar"

ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${JAVA_HOME}/bin:${PATH}"

# Layer 4: dependency aplikasi (sering berubah, ditaruh paling akhir)
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Layer 5: PyTorch (CPU-only) untuk model deep learning Fase 4
# Dipasang dari indeks CPU PyTorch: wheel-nya ~200 MB, sedangkan wheel default
# di PyPI membawa dependensi CUDA ~2,5 GB yang sama sekali tidak terpakai di
# lingkungan tanpa GPU ini. --retries dinaikkan karena unduhan besar di jaringan
# tidak stabil adalah kegagalan build yang paling sering terjadi di proyek ini.
RUN pip install --retries 10 --timeout 120 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1

# Layer 6: encoder kalimat pralatih (BGE)
# Bobotnya di-bake ke dalam image dengan alasan yang sama seperti jar S3A di
# atas: jalur inferensi tidak boleh bergantung pada koneksi ke Hugging Face.
# Artefak joblib hanya menyimpan NAMA model, jadi tanpa langkah ini container
# yang tanpa jaringan akan gagal memuat Head A arsitektur `hybrid`.
#
# HF_HOME ditaruh di luar $HOME supaya cache-nya dapat dibaca semua user
# container (build berjalan sebagai root, runtime sebagai `spark`). Izinnya
# harus mengandung WRITE, bukan sekadar baca: huggingface_hub menulis berkas
# penanda (`.no_exist`, lock) ke dalam cache bahkan ketika seluruh bobot sudah
# lengkap, sehingga cache read-only mematikan inferensi dengan PermissionError.
#
# `version.txt` ditulis di sini karena transformers menganggap cache tanpa berkas
# itu sebagai cache pra-v4.22 yang perlu dimigrasi, lalu memuntahkan peringatan
# "semua panggilan from_pretrained() kemungkinan besar akan gagal" — keliru, dan
# persis jenis umpan yang bikin orang mengejar bug yang tidak ada.
# Unduhannya dibungkus retry: registry HF memutus koneksi di tengah berkas
# jauh lebih sering daripada yang diduga, dan snapshot yang terpotong baru
# ketahuan saat inferensi pertama.
ENV HF_HOME=/opt/hf-cache \
    SENTENCE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
COPY fetch_encoder.py /tmp/fetch_encoder.py
RUN mkdir -p ${HF_HOME} \
    && python /tmp/fetch_encoder.py \
    && echo 1 > ${HF_HOME}/hub/version.txt \
    && chmod -R a+rwX ${HF_HOME}

# POS tagger NLTK untuk ide judul AI Studio (src/common/title_templates.py):
# membedakan kata benda/sifat/kerja di cerita pengguna dan di judul katalog.
# Tanpa data ini generator tetap jalan, hanya tanpa pemeriksaan jenis kata.
ENV NLTK_DATA=/opt/nltk_data
RUN python -c "import nltk, sys; sys.exit(0 if nltk.download('averaged_perceptron_tagger_eng', download_dir='/opt/nltk_data', quiet=True) else 1)" \
    && chmod -R a+rX /opt/nltk_data

# Seluruh bobot sudah berada di dalam image, jadi jalur inferensi tidak perlu
# lagi menghubungi Hugging Face — itulah alasan Layer 6 ada. Tanpa flag ini
# huggingface_hub tetap melakukan panggilan revalidasi tiap kali encoder dimuat:
# menambah latensi saat online, dan gagal total saat tidak ada jaringan.
# Disetel SETELAH unduhan, karena menyetelnya lebih awal membuat
# fetch_encoder.py sendiri tidak bisa mengambil bobotnya.
ENV HF_HUB_OFFLINE=1

# Layer 7: workspace
WORKDIR /opt/project
RUN useradd --create-home --uid 1000 spark \
    && mkdir -p /opt/project /home/spark/.ivy2 \
    && chown -R spark:spark /opt/project /home/spark

USER spark

CMD ["bash"]

# Layer 6: pembatasan thread BLAS/OpenMP
# numpy/scikit-learn (OpenBLAS) mendeteksi seluruh core host dan membuka
# thread-pool sebesar itu. Di dalam container yang juga menjalankan JVM Spark,
# kombinasi ini bisa deadlock: seluruh thread berhenti di futex_wait dan job
# menggantung tanpa pesan error. Membatasinya secara eksplisit menghilangkan
# deadlock tersebut, dan pada dataset sebesar ini tidak menurunkan kecepatan
# secara berarti.
ENV OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2
