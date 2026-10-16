# -*- coding: utf-8 -*-
"""Terjemahkan streamlit_app.py + model_registry.py ke bahasa Inggris.

Memakai pencocokan string persis, bukan tulis-ulang berkas, supaya tidak ada
satu baris logika pun yang bisa bergeser. Setiap pasangan yang tidak ketemu
dilaporkan sebagai galat dan seluruh proses dibatalkan.
"""
import sys

APP = "/opt/project/app/streamlit_app.py"
REG = "/opt/project/src/common/model_registry.py"

Q = '"'
S = "'"

APP_PAIRS = [
    # docstring modul
    ('''"""
NovelLens — antarmuka produk
============================
Aplikasi Streamlit yang menyajikan hasil pipeline sebagai produk siap pakai,
bukan panel eksperimen. Konsekuensinya pada desain:

  * Tidak ada detail infrastruktur (endpoint storage, jumlah objek per bucket,
    daftar centang fase pipeline). Itu urusan operator, bukan pengguna.
  * Kegagalan backend ditampilkan sebagai pesan manusiawi; jejak teknisnya
    disembunyikan di balik expander untuk pemelihara.
  * Angka performa model tetap ditampilkan — di halaman Metodologi — karena
    pengguna berhak tahu seberapa jauh keluaran AI boleh dipercaya. Yang
    dibuang adalah artefak proses coba-coba, bukan transparansinya.

Data dibaca dari hasil kueri Spark SQL yang sudah dimaterialisasi (Fase 3);
tidak ada Spark yang dijalankan dari sini agar interaksi tetap instan.
"""''',
     '''"""
NovelLens — product interface
=============================
A Streamlit app that presents the pipeline output as a finished product rather
than an experiment console. That choice drives the design:

  * No infrastructure details (storage endpoints, per-bucket object counts,
    pipeline-phase checklists). That is the operator's concern, not the user's.
  * Backend failures surface as plain-language messages; the technical trace is
    tucked behind an expander for maintainers.
  * Model performance numbers stay visible — on the Methodology page — because
    users deserve to know how far the AI output can be trusted. What was cut is
    the trial-and-error residue, not the transparency.

Data is read from materialised Spark SQL results (Phase 3); no Spark job runs
from here, so interaction stays instant.
"""'''),

    # konfigurasi halaman & CSS
    ('page_title="NovelLens — Intelijen Katalog Novel"',
     'page_title="NovelLens — Novel Catalogue Intelligence"'),
    ("# --- Identitas visual ---", "# --- Visual identity ---"),
    ("/* --- Ritme halaman ---", "/* --- Page rhythm ---"),
    ("/* Skala tipografi yang konsisten, bukan ukuran acak per elemen. */",
     "/* A consistent type scale, not a per-element size free-for-all. */"),
    ("/* --- Kartu metrik ---", "/* --- Metric cards ---"),
    ("/* --- Chip genre ---", "/* --- Genre chips ---"),
    ("/* --- Teks pendukung ---", "/* --- Supporting text ---"),
    ("/* --- Tab, tombol, tabel ---", "/* --- Tabs, buttons, tables ---"),

    # lapisan data
    ("# Lapisan data", "# Data layer"),
    ('"""Data produk belum siap disajikan (pipeline belum dijalankan)."""',
     '"""Product data is not ready to serve yet (the pipeline has not run)."""'),
    ('"""Baca satu tabel hasil analitik dari data lake."""',
     '"""Read one analytics result table from the data lake."""'),
    ('"""Cek ketersediaan penyimpanan data tanpa membocorkan detailnya ke UI."""',
     '"""Check storage availability without leaking its details into the UI."""'),

    # komponen UI
    ("# Komponen UI", "# UI components"),
    ('"""Kepala halaman: label kecil, judul, satu kalimat penjelas."""',
     '"""Page header: eyebrow label, title, one explanatory sentence."""'),
    ('"""Samakan gaya seluruh grafik dengan tema aplikasi."""',
     '"""Match every chart to the app theme."""'),
    ('"""Kartu hasil rekomendasi: judul, genre, cuplikan sinopsis, metrik."""',
     '"""Recommendation card: title, genres, synopsis snippet, metrics."""'),
    ('st.caption(f"Kemiripan {row[' + S + 'similarity' + S + ']:.0%}")',
     'st.caption(f"Similarity {row[' + S + 'similarity' + S + ']:.0%}")'),
    ('meta.append(f"{int(row[' + S + 'chapter_count' + S + ']):,} bab")',
     'meta.append(f"{int(row[' + S + 'chapter_count' + S + ']):,} chapters")'),
    ('meta.append(f"{int(row[' + S + 'rating_votes' + S + ']):,} suara")',
     'meta.append(f"{int(row[' + S + 'rating_votes' + S + ']):,} votes")'),
    ('cols[1].link_button("Buka", row["url"], use_container_width=True)',
     'cols[1].link_button("Open", row["url"], use_container_width=True)'),

    # halaman ringkasan
    ("# HALAMAN — Ringkasan", "# PAGE — Overview"),
    ('''        "Intelijen katalog untuk 4.461 judul web novel: apa yang dibaca orang, "
        "apa yang mereka nilai tinggi, dan judul apa yang sebaiknya direkomendasikan.",
        eyebrow="Ringkasan katalog",''',
     '''        "Catalogue intelligence for 4,461 web novel titles: what people read, "
        "what they rate highly, and which titles are worth recommending.",
        eyebrow="Catalogue overview",'''),
    ('''card(cols[0], "Judul dalam katalog", f"{len(viz):,}", f"{viz['primary_genre'].nunique()} genre utama")''',
     '''card(cols[0], "Titles in catalogue", f"{len(viz):,}", f"{viz['primary_genre'].nunique()} primary genres")'''),
    ('''    card(cols[1], "Rating rata-rata", f"{viz['weighted_rating'].mean():.2f}",
         f"{len(reliable):,} judul dengan penilaian mapan")
    card(cols[2], "Median panjang", f"{viz['chapter_count'].median():,.0f} bab",
         f"terpanjang {viz['chapter_count'].max():,.0f} bab")
    card(cols[3], "Genre teratas", genres.iloc[0]["genre"],
         f"{genres.iloc[0]['pct_of_catalog']:.0f}% dari katalog")''',
     '''    card(cols[1], "Average rating", f"{viz['weighted_rating'].mean():.2f}",
         f"{len(reliable):,} titles with an established score")
    card(cols[2], "Median length", f"{viz['chapter_count'].median():,.0f} chapters",
         f"longest {viz['chapter_count'].max():,.0f} chapters")
    card(cols[3], "Top genre", genres.iloc[0]["genre"],
         f"{genres.iloc[0]['pct_of_catalog']:.0f}% of the catalogue")'''),
    ('st.subheader("Yang perlu diketahui")', 'st.subheader("What you need to know")'),
    ('''        insight_cols[0], "Panjang ≠ disukai", f"r = {r_len}",
        "Korelasi jumlah bab dengan rating justru negatif tipis. Menambah bab "
        "tidak membeli penerimaan pembaca.",''',
     '''        insight_cols[0], "Length ≠ acclaim", f"r = {r_len}",
        "Chapter count correlates slightly negatively with rating. Adding more "
        "chapters does not buy reader approval.",'''),
    ('''        insight_cols[1], "Genre paling dihargai", top_genre["genre"],
        f"Rata-rata {top_genre['avg_weighted_rating']:.2f} dari {int(top_genre['n_novels'])} judul. "
        f"Terpanjang: {longest['genre']} (~{longest['avg_chapters']:,.0f} bab).",''',
     '''        insight_cols[1], "Best-rated genre", top_genre["genre"],
        f"Averages {top_genre['avg_weighted_rating']:.2f} across {int(top_genre['n_novels'])} titles. "
        f"Longest: {longest['genre']} (~{longest['avg_chapters']:,.0f} chapters).",'''),
    ('''        insight_cols[2], "Sinopsis yang kaya menang", f"+{gain:.2f} poin",
        "Judul dengan sinopsis 300+ kata dinilai lebih tinggi dibanding yang "
        "di bawah 50 kata.",''',
     '''        insight_cols[2], "Rich synopses win", f"+{gain:.2f} points",
        "Titles with 300+ word synopses score higher than those under "
        "50 words.",'''),
    ('st.subheader("Genre terpopuler")', 'st.subheader("Most common genres")'),
    ('st.subheader("Paling dihargai pembaca")', 'st.subheader("Highest reader ratings")'),
    ('.rename(columns={"title": "Judul", "weighted_rating": "Rating", "rating_votes": "Suara"})',
     '.rename(columns={"title": "Title", "weighted_rating": "Rating", "rating_votes": "Votes"})'),

    # halaman analitik
    ("# HALAMAN — Analitik Katalog", "# PAGE — Catalogue Analytics"),
    ('''        "Analitik katalog",
        "Struktur pasar novel terjemahan: komposisi genre, hubungan panjang cerita "
        "dengan penerimaan pembaca, dan kosakata yang mendominasi ceritanya.",
        eyebrow="Business intelligence",''',
     '''        "Catalogue analytics",
        "The structure of the translated-novel market: genre mix, how story length "
        "relates to reader reception, and the vocabulary that dominates the stories.",
        eyebrow="Business intelligence",'''),
    ('["Genre", "Panjang & Rating", "Bahasa Cerita", "Peta Pasar"]',
     '["Genres", "Length & Rating", "Story Language", "Market Map"]'),
    ("    # --- Genre ---", "    # --- Genres ---"),
    ('st.slider("Genre ditampilkan", 5, min(30, len(genres)), 15)',
     'st.slider("Genres shown", 5, min(30, len(genres)), 15)'),
    ('''            '<div class="nl-note">Panjang batang = jumlah judul, warna = rating rata-rata. '
            "Genre terbanyak belum tentu genre yang paling disukai — bandingkan keduanya.</div>",''',
     '''            '<div class="nl-note">Bar length = number of titles, colour = average rating. '
            "The most common genre is not necessarily the best-liked one — compare both.</div>",'''),
    ('st.markdown("**Kombinasi genre tersering**")', 'st.markdown("**Most frequent genre combinations**")'),
    ('''                    columns={"genre_combo": "Kombinasi", "n_novels": "Judul",
                             "avg_weighted_rating": "Rating"}),''',
     '''                    columns={"genre_combo": "Combination", "n_novels": "Titles",
                             "avg_weighted_rating": "Rating"}),'''),
    ('st.markdown("**Sebaran rating per genre**")', 'st.markdown("**Rating spread by genre**")'),
    ("    # --- Panjang & rating ---", "    # --- Length & rating ---"),
    ('picked = col_a.multiselect("Genre", available,', 'picked = col_a.multiselect("Genres", available,'),
    ('only_reliable = col_b.checkbox("Hanya judul dengan penilaian mapan", value=True)',
     'only_reliable = col_b.checkbox("Only titles with an established score", value=True)'),
    ('st.info("Tidak ada judul yang cocok dengan filter ini.")', 'st.info("No titles match these filters.")'),
    ('''                labels={"chapter_count": "Jumlah bab (skala log)",
                        "weighted_rating": "Rating", "primary_genre": ""},''',
     '''                labels={"chapter_count": "Chapter count (log scale)",
                        "weighted_rating": "Rating", "primary_genre": ""},'''),
    ('mode="lines+markers", name="Rata-rata per kelompok panjang",',
     'mode="lines+markers", name="Average per length bucket",'),
    ('''                f'<div class="nl-note">Garis oranye adalah rata-rata rating per kelompok panjang. '
                f"Korelasinya r = {r} — praktis datar lalu menurun di atas 1.000 bab: cerita yang "
                "terlalu panjang cenderung kehilangan pembaca yang puas.</div>",''',
     '''                f'<div class="nl-note">The orange line is the average rating per length bucket. '
                f"The correlation is r = {r} — essentially flat, then falling past 1,000 chapters: "
                "stories that run too long tend to lose satisfied readers.</div>",'''),
    ('st.markdown("**Rating menurut kelompok panjang**")', 'st.markdown("**Rating by length bucket**")'),
    ("    # --- Bahasa cerita ---", "    # --- Story language ---"),
    ('st.markdown("**Kosakata yang mendominasi sinopsis**")',
     'st.markdown("**Vocabulary that dominates the synopses**")'),
    ('                    columns={"token": "Kata", "n_novels": "Judul"}),',
     '                    columns={"token": "Word", "n_novels": "Titles"}),'),
    ('''            '<div class="nl-note">Diambil dari sinopsis asli (bukan tag katalog), '
            "setelah pembuangan stopword.</div>",''',
     '''            '<div class="nl-note">Taken from the original synopses (not the catalogue tags), '
            "after stopword removal.</div>",'''),
    ('st.markdown("**Panjang sinopsis vs rating**")', 'st.markdown("**Synopsis length vs rating**")'),
    ('st.markdown("**Elemen cerita dengan rating terbaik**")', 'st.markdown("**Best-rated story elements**")'),
    ("    # --- Peta pasar ---", "    # --- Market map ---"),
    ('st.markdown("**Rating rata-rata per genre × negara asal**")',
     'st.markdown("**Average rating by genre × country of origin**")'),
    ('labels=dict(x="Negara asal", y="", color="Rating"),',
     'labels=dict(x="Country of origin", y="", color="Rating"),'),
    ('''            '<div class="nl-note">Sel kosong = kombinasi dengan kurang dari 25 judul, '
            "sengaja tidak dihitung agar rata-rata tidak dibentuk segelintir judul.</div>",''',
     '''            '<div class="nl-note">Empty cells = combinations with fewer than 25 titles, '
            "deliberately left uncomputed so no average rests on a handful of titles.</div>",'''),
    ('st.markdown("**Tren per dekade**")', 'st.markdown("**Trend by decade**")'),
    ('st.markdown("**Judul terpanjang di katalog**")', 'st.markdown("**Longest titles in the catalogue**")'),
    ('''                    columns={"title": "Judul", "chapter_count": "Bab", "rating_score": "Rating"}),''',
     '''                    columns={"title": "Title", "chapter_count": "Chapters", "rating_score": "Rating"}),'''),
    ('"""Render word cloud dengan latar transparan agar menyatu dengan tema."""',
     '"""Render the word cloud on a transparent ground so it blends with the theme."""'),

    # AI Studio
    ("# HALAMAN — AI Studio", "# PAGE — AI Studio"),
    ('''CONTOH_SINOPSIS = (
    "Setelah dikhianati sekutunya dan tewas di puncak menara, seorang pendekar "
    "terbangun kembali di tubuh dirinya sepuluh tahun lebih muda. Kali ini ia "
    "berniat menguasai teknik pedang terlarang sebelum musuh lamanya bangkit."
)
CONTOH_TAG = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"''',
     '''SAMPLE_SYNOPSIS = (
    "After being betrayed by his allies and dying at the top of the tower, a "
    "swordsman wakes up in his own body ten years younger. This time he intends "
    "to master the forbidden sword techniques before his old enemy rises again."
)
SAMPLE_TAGS = "Time Travel, Weak to Strong, Cultivation, Male Protagonist, Revenge, Sword Wielder"'''),
    ("CONTOH_SINOPSIS", "SAMPLE_SYNOPSIS"),
    ("CONTOH_TAG", "SAMPLE_TAGS"),
    ('''        "Empat asisten yang bekerja dari isi cerita: mencari judul serupa, "
        "melabeli genre, memperkirakan penerimaan pembaca, dan mengenali arketipe naratif.",
        eyebrow="Model prediktif",''',
     '''        "Four assistants that work from story content: finding similar titles, "
        "tagging genres, estimating reader reception, and recognising narrative archetypes.",
        eyebrow="Predictive models",'''),
    ('st.info("Fitur AI sedang disiapkan. Coba lagi setelah proses pelatihan model selesai.")',
     'st.info("AI features are still being prepared. Try again once model training has finished.")'),
    ('st.info("Fitur AI belum tersedia.")', 'st.info("AI features are not available yet.")'),
    ('with st.expander("Detail teknis"):', 'with st.expander("Technical details"):'),
    ('["Rekomendasi", "Pelabelan Genre", "Perkiraan Rating", "Arketipe Cerita"]',
     '["Recommendations", "Genre Tagging", "Rating Estimate", "Story Archetypes"]'),
    ('st.markdown("Temukan judul yang secara isi paling dekat dengan yang Anda sukai.")',
     'st.markdown("Find the titles whose content sits closest to one you already like.")'),
    ('mode = st.radio("Cari dari", ["Judul di katalog", "Deskripsi bebas"], horizontal=True)',
     'mode = st.radio("Search by", ["Catalogue title", "Free-text description"], horizontal=True)'),
    ('k = st.slider("Jumlah rekomendasi", 3, 12, 6)', 'k = st.slider("How many recommendations", 3, 12, 6)'),
    ('if mode == "Judul di katalog":', 'if mode == "Catalogue title":'),
    ('choice = st.selectbox("Judul acuan", titles)', 'choice = st.selectbox("Reference title", titles)'),
    ('text = st.text_area("Cerita seperti apa yang Anda cari?", SAMPLE_SYNOPSIS, height=110)',
     'text = st.text_area("What kind of story are you looking for?", SAMPLE_SYNOPSIS, height=110)'),
    ('st.warning("Masukan belum cukup untuk dicarikan padanan. Tambahkan detail cerita.")',
     'st.warning("Not enough input to find a match. Add more story detail.")'),
    ('''        "Tempel sinopsis dan tag sebuah novel, lalu pilih model yang ingin dipakai. "
        "Setiap model punya cara berbeda dalam menimbang kata."''',
     '''        "Paste a novel's synopsis and tags, then pick the model to use. "
        "Each model weighs words differently."'''),
    ('''    choice = st.radio(
        "Model prediksi",''',
     '''    choice = st.radio(
        "Prediction model",'''),
    ('+ (f\'<br><span style="opacity:.75">Arsitektur mengikuti {reference}.</span>\' if reference else "")',
     '+ (f\'<br><span style="opacity:.75">Architecture follows {reference}.</span>\' if reference else "")'),
    ('synopsis = st.text_area("Sinopsis", SAMPLE_SYNOPSIS, height=150)',
     'synopsis = st.text_area("Synopsis", SAMPLE_SYNOPSIS, height=150)'),
    ('tags = st.text_input("Tag (opsional, pisahkan dengan koma)", SAMPLE_TAGS)',
     'tags = st.text_input("Tags (optional, comma-separated)", SAMPLE_TAGS)'),
    ('origin = st.selectbox("Negara asal", ["CN", "KR", "JP", "EN", "Other"])',
     'origin = st.selectbox("Country of origin", ["CN", "KR", "JP", "EN", "Other"])'),
    ('novel_type = st.selectbox("Tipe", ["Web Novel", "Light Novel", "Published Novel"])',
     'novel_type = st.selectbox("Format", ["Web Novel", "Light Novel", "Published Novel"])'),
    ('''        sensitivity = st.slider(
            "Ketelitian", 0.6, 1.6, 1.0, 0.05,
            help="Di bawah 1 menampilkan lebih banyak genre; di atas 1 hanya yang paling yakin.",
        )''',
     '''        sensitivity = st.slider(
            "Strictness", 0.6, 1.6, 1.0, 0.05,
            help="Below 1 surfaces more genres; above 1 keeps only the most confident ones.",
        )'''),
    ('if st.button("Prediksi genre", type="primary"):', 'if st.button("Predict genres", type="primary"):'),
    ('st.warning("Teks belum cukup untuk dianalisis. Tambahkan sinopsis atau tag.")',
     'st.warning("Not enough text to analyse. Add a synopsis or some tags.")'),
    ('predicted = result[result["diprediksi"]]["genre"].tolist()',
     'predicted = result[result["predicted"]]["genre"].tolist()'),
    ('st.markdown("**Genre terprediksi**")', 'st.markdown("**Predicted genres**")'),
    ('st.info("Tidak ada genre yang cukup meyakinkan. Turunkan ketelitian atau tambah detail.")',
     'st.info("No genre is confident enough. Lower the strictness or add more detail.")'),
    ('''            color="diprediksi", color_discrete_map={True: ACCENT, False: "rgba(255,255,255,.18)"},
            labels={"probability": "Keyakinan model", "genre": "", "diprediksi": ""},''',
     '''            color="predicted", color_discrete_map={True: ACCENT, False: "rgba(255,255,255,.18)"},
            labels={"probability": "Model confidence", "genre": "", "predicted": ""},'''),
    ('name="Ambang genre",', 'name="Genre threshold",'),
    ('''            '<div class="nl-note">Garis oranye adalah ambang khusus tiap genre. '
            "Genre langka punya ambang lebih rendah, sehingga tetap bisa terdeteksi "
            "tanpa membanjiri hasil dengan genre umum.</div>",''',
     '''            '<div class="nl-note">The orange marks are each genre&apos;s own threshold. '
            "Rare genres get a lower one, so they can still be detected without "
            "flooding the results with common genres.</div>",'''),
    ('st.info("Perkiraan rating belum tersedia.")', 'st.info("Rating estimation is not available yet.")'),
    ('''        "Perkirakan bagaimana pembaca akan menilai sebuah cerita — berguna sebelum "
        "judul baru punya rating sungguhan."''',
     '''        "Estimate how readers will score a story — useful before a new title has "
        "any real ratings."'''),
    ('text = st.text_area("Sinopsis atau tag", SAMPLE_TAGS, height=130, key="rating_text")',
     'text = st.text_area("Synopsis or tags", SAMPLE_TAGS, height=130, key="rating_text")'),
    ('chapters = st.number_input("Rencana jumlah bab", 1, 12000, 300, 10)',
     'chapters = st.number_input("Planned chapter count", 1, 12000, 300, 10)'),
    ('year = st.number_input("Tahun rilis", 1990, 2030, 2024)',
     'year = st.number_input("Release year", 1990, 2030, 2024)'),
    ('origin = st.selectbox("Negara asal", ["CN", "KR", "JP", "EN", "Other"], key="rating_origin")',
     'origin = st.selectbox("Country of origin", ["CN", "KR", "JP", "EN", "Other"], key="rating_origin")'),
    ('completed = st.checkbox("Sudah tamat")', 'completed = st.checkbox("Already completed")'),
    ('if st.button("Perkirakan rating", type="primary"):', 'if st.button("Estimate rating", type="primary"):'),
    ('st.warning("Masukan belum cukup untuk diperkirakan.")',
     'st.warning("Not enough input to produce an estimate.")'),
    ('''        card(cols[0], "Perkiraan rating", f"{predicted:.2f}", "skala 1–5")
        card(cols[1], "Dibanding katalog", f"{predicted - average:+.2f}", f"rata-rata {average:.2f}")
        card(cols[2], "Rentang kesalahan", f"±{head_d['mae']:.2f}", "berdasarkan uji tertahan")''',
     '''        card(cols[0], "Estimated rating", f"{predicted:.2f}", "on a 1–5 scale")
        card(cols[1], "Vs catalogue", f"{predicted - average:+.2f}", f"average {average:.2f}")
        card(cols[2], "Error margin", f"±{head_d['mae']:.2f}", "measured on the held-out test set")'''),
    ('''            '<div class="nl-note">Perkiraan ini hanya membaca isi cerita, bukan kualitas '
            "penulisan atau promosi. Konten menjelaskan sebagian kecil variasi rating, jadi "
            "perlakukan angkanya sebagai indikasi arah — bukan vonis.</div>",''',
     '''            '<div class="nl-note">This estimate reads story content only — not writing quality '
            "or marketing. Content explains a small share of rating variance, so treat the "
            "number as a direction, not a verdict.</div>",'''),
    ('''        f"Katalog terbagi menjadi {head_c['best_k']} arketipe naratif. "
        "Tempel sebuah cerita untuk melihat ia masuk keluarga mana."''',
     '''        f"The catalogue splits into {head_c['best_k']} narrative archetypes. "
        "Paste a story to see which family it belongs to."'''),
    ('text = st.text_area("Sinopsis atau tag", SAMPLE_TAGS, height=110, key="arch_text")',
     'text = st.text_area("Synopsis or tags", SAMPLE_TAGS, height=110, key="arch_text")'),
    ('if st.button("Kenali arketipe", type="primary"):', 'if st.button("Identify archetype", type="primary"):'),
    ('st.warning("Teks belum cukup untuk dikenali.")', 'st.warning("Not enough text to identify an archetype.")'),
    ('st.markdown(f"### Arketipe #{assignment[' + S + 'cluster_id' + S + ']}")',
     'st.markdown(f"### Archetype #{assignment[' + S + 'cluster_id' + S + ']}")'),
    ('''            card(cols[0], "Judul sekeluarga", f"{profile.get('size', 0):,}")
            card(cols[1], "Rating rata-rata", f"{profile.get('avg_weighted_rating', 0):.2f}")
            card(cols[2], "Panjang rata-rata", f"{profile.get('avg_chapters', 0):,.0f} bab")''',
     '''            card(cols[0], "Titles in this family", f"{profile.get('size', 0):,}")
            card(cols[1], "Average rating", f"{profile.get('avg_weighted_rating', 0):.2f}")
            card(cols[2], "Average length", f"{profile.get('avg_chapters', 0):,.0f} chapters")'''),
    ('st.caption("Contoh judul: " + " · ".join(profile.get("example_titles", [])[:4]))',
     'st.caption("Example titles: " + " · ".join(profile.get("example_titles", [])[:4]))'),
    ('st.markdown("**Seluruh arketipe**")', 'st.markdown("**All archetypes**")'),
    ('''    overview = profiles.assign(
        istilah=profiles["top_terms"].apply(lambda t: ", ".join(t[:5])),
        genre=profiles["top_genres"].apply(lambda g: ", ".join(list(g.keys())[:3])),
    )[["cluster_id", "size", "avg_weighted_rating", "avg_chapters", "istilah", "genre"]]
    st.dataframe(
        overview.rename(columns={
            "cluster_id": "#", "size": "Judul", "avg_weighted_rating": "Rating",
            "avg_chapters": "Bab", "istilah": "Ciri khas", "genre": "Genre dominan"}),''',
     '''    overview = profiles.assign(
        terms=profiles["top_terms"].apply(lambda t: ", ".join(t[:5])),
        genre=profiles["top_genres"].apply(lambda g: ", ".join(list(g.keys())[:3])),
    )[["cluster_id", "size", "avg_weighted_rating", "avg_chapters", "terms", "genre"]]
    st.dataframe(
        overview.rename(columns={
            "cluster_id": "#", "size": "Titles", "avg_weighted_rating": "Rating",
            "avg_chapters": "Chapters", "terms": "Signature terms", "genre": "Dominant genres"}),'''),

    # halaman metodologi
    ("# HALAMAN — Metodologi", "# PAGE — Methodology"),
    ('''        "Metodologi & keterbatasan",
        "Bagaimana angka di aplikasi ini dihitung, seberapa akurat modelnya, "
        "dan di titik mana hasilnya tidak boleh dipercaya mentah-mentah.",
        eyebrow="Transparansi",''',
     '''        "Methodology & limitations",
        "How the numbers in this app are computed, how accurate the models are, "
        "and where the output should not be taken at face value.",
        eyebrow="Transparency",'''),
    ('st.subheader("Dari mana datanya")', 'st.subheader("Where the data comes from")'),
    ('''Katalog berisi **4.461 judul** web novel terjemahan beserta metadata, rating
pembaca, dan sinopsis. Sebelum dipakai, data melewati pembersihan:

- Nilai kosong yang disamarkan sebagai angka `0` atau teks `"Unknown"`
  dikembalikan menjadi kosong sungguhan — tanpa ini rata-rata jumlah bab
  meleset sekitar 20%.
- Rating ditampilkan dalam bentuk **rating tertimbang**: judul dengan 2 suara
  tidak diperlakukan sama dengan judul dengan 500 suara. Peringkat apa pun di
  aplikasi ini hanya melibatkan judul dengan minimal 10 suara.
- Sinopsis digabungkan dari sumber terpisah lewat pencocokan tautan, mencakup
  **98,45%** katalog.''',
     '''The catalogue holds **4,461 translated web novel titles** with their metadata,
reader ratings, and synopses. Before use, the data goes through cleaning:

- Missing values disguised as the number `0` or the text `"Unknown"` are
  restored to genuinely empty — without this the average chapter count is off
  by roughly 20%.
- Ratings are shown as a **weighted rating**: a title with 2 votes is not
  treated like one with 500. Every ranking in this app uses only titles with at
  least 10 votes.
- Synopses are joined in from a separate source by URL matching, covering
  **98.45%** of the catalogue.'''),
    ('st.subheader("Performa model pelabelan genre")', 'st.subheader("Genre tagging model performance")'),
    ('''        "Ketiga model dilatih dan diuji pada pembagian data yang sama "
        f"({head_a['n_train']:,} judul untuk pengembangan · {head_a['n_test']:,} untuk pengujian), "
        "sehingga perbandingannya adil. Hyperparameter dan ambang keputusan tiap genre "
        "dipilih memakai validasi silang di dalam set pengembangan — set uji baru "
        "disentuh sekali, di akhir, agar angka di bawah ini tidak dilebih-lebihkan."''',
     '''        "Every model is trained and tested on the same split "
        f"({head_a['n_train']:,} titles for development · {head_a['n_test']:,} for testing), "
        "so the comparison is fair. Hyperparameters and each genre's decision threshold "
        "are chosen by cross-validation inside the development set — the test set is "
        "touched once, at the very end, so the numbers below are not inflated."'''),
    ('''        rows.append({
            "Model": entry.get("label", name),
            "Pendekatan": entry.get("family", ""),
            "F1 makro": entry["f1_macro"],
            "F1 mikro": entry["f1_micro"],
            "P@3": entry.get("precision_at_3"),
            "Hamming loss": entry.get("hamming_loss"),
            "F1 makro tanpa tuning ambang": entry["f1_macro_fixed_threshold"],
        })
    table = pd.DataFrame(rows).sort_values("F1 makro", ascending=False)''',
     '''        rows.append({
            "Model": entry.get("label", name),
            "Approach": entry.get("family", ""),
            "Macro F1": entry["f1_macro"],
            "Micro F1": entry["f1_micro"],
            "P@3": entry.get("precision_at_3"),
            "Hamming loss": entry.get("hamming_loss"),
            "Macro F1 without threshold tuning": entry["f1_macro_fixed_threshold"],
        })
    table = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)'''),
    ('''            "F1 makro": st.column_config.ProgressColumn("F1 makro", min_value=0, max_value=1, format="%.3f"),
            "F1 mikro": st.column_config.ProgressColumn("F1 mikro", min_value=0, max_value=1, format="%.3f"),''',
     '''            "Macro F1": st.column_config.ProgressColumn("Macro F1", min_value=0, max_value=1, format="%.3f"),
            "Micro F1": st.column_config.ProgressColumn("Micro F1", min_value=0, max_value=1, format="%.3f"),'''),
    ('''<b>F1 makro</b> memberi bobot sama ke tiap genre, jadi turun tajam bila genre langka
diabaikan — ini metrik pemilihan model. <b>F1 mikro</b> didominasi genre umum.
<b>P@3</b> mengukur ketepatan tiga tebakan teratas tanpa bergantung ambang sama sekali,
yang paling dekat dengan apa yang dilihat pengguna. <b>Hamming loss</b> adalah proporsi
keputusan biner yang salah (makin kecil makin baik). Kolom terakhir memperlihatkan skor
model yang sama bila memakai ambang seragam 0,5 — selisihnya adalah kontribusi
penyetelan ambang per genre.''',
     '''<b>Macro F1</b> weights every genre equally, so it drops sharply when rare genres are
ignored — this is the model selection metric. <b>Micro F1</b> is dominated by common
genres. <b>P@3</b> measures how accurate the top three guesses are without depending on a
threshold at all, which is closest to what users actually see. <b>Hamming loss</b> is the
share of binary decisions that are wrong (lower is better). The last column shows the same
model scored with a uniform 0.5 threshold — the gap is the contribution of per-genre
threshold tuning.'''),
    ('''**Rujukan arsitektur.** Dua model berbasis fitur (TF-IDF + linear) mengikuti baseline
yang lazim pada penelitian klasifikasi genre dari sinopsis. Model urutan mereplikasi
BiLSTM dengan self-attention (Ertugrul & Karagoz, 2018; Zhou et al., 2016),
dengan perlakuan multi-label *binary cross-entropy* per label berbobot
(Liu et al., SIGIR 2017). Penyetelan ambang probabilitas per genre mengikuti
Hoang (2018).''',
     '''**Architecture references.** The two feature-based models (TF-IDF + linear) follow the
baselines common in genre-classification-from-synopsis research. The sequence model
replicates a BiLSTM with self-attention (Ertugrul & Karagoz, 2018; Zhou et al., 2016),
with multi-label handling via weighted per-label *binary cross-entropy*
(Liu et al., SIGIR 2017). Per-genre probability threshold tuning follows Hoang (2018).'''),
    ('st.subheader("Fitur lain")', 'st.subheader("Other features")'),
    ('''    card(cols[0], "Rekomendasi", f"{head_b['genre_overlap_at_k']:.0%}",
         f"dari {head_b['k']} saran berbagi genre dengan judul acuan")
    card(cols[1], "Arketipe", f"{head_c['best_k']} kelompok",
         "jumlah kelompok dipilih otomatis lewat silhouette score")
    card(cols[2], "Perkiraan rating", f"±{head_d['mae']:.2f}",
         f"lebih baik {head_d['improvement_over_baseline_pct']:.0f}% dari tebakan rata-rata")''',
     '''    card(cols[0], "Recommendations", f"{head_b['genre_overlap_at_k']:.0%}",
         f"of {head_b['k']} suggestions share a genre with the reference title")
    card(cols[1], "Archetypes", f"{head_c['best_k']} clusters",
         "cluster count chosen automatically by silhouette score")
    card(cols[2], "Rating estimate", f"±{head_d['mae']:.2f}",
         f"{head_d['improvement_over_baseline_pct']:.0f}% better than guessing the average")'''),
    ('st.subheader("Keterbatasan yang perlu diingat")', 'st.subheader("Limitations worth remembering")'),
    ('''- Model membaca **deskripsi**, bukan naskah. Novel dengan sinopsis buruk akan
  dinilai buruk meski isinya bagus.
- Rekomendasi murni berbasis kemiripan konten; dua judul bertema sama dianggap
  mirip walau mutu penulisannya berbeda jauh.
- Katalog didominasi novel Tiongkok (55%), sehingga akurasi untuk judul
  berbahasa Inggris atau Jepang cenderung lebih rendah.
- Rating berasal dari satu komunitas pembaca terjemahan, bukan populasi umum.
- Arketipe adalah kelompok yang saling bersinggungan, bukan sekat tegas.''',
     '''- The models read the **description**, not the manuscript. A novel with a poor
  synopsis will be judged poorly even if the story itself is good.
- Recommendations rest purely on content similarity; two titles on the same theme
  count as similar even when their writing quality differs sharply.
- The catalogue is dominated by Chinese novels (55%), so accuracy on English or
  Japanese titles tends to be lower.
- Ratings come from a single translated-fiction reader community, not the general
  population.
- Archetypes are overlapping groups, not hard partitions.'''),

    # router & sidebar
    ('''PAGES = {
    "Ringkasan": page_overview,
    "Analitik Katalog": page_analytics,
    "AI Studio": page_studio,
    "Metodologi": page_method,
}''',
     '''PAGES = {
    "Overview": page_overview,
    "Catalogue Analytics": page_analytics,
    "AI Studio": page_studio,
    "Methodology": page_method,
}'''),
    ('f\'<div style="font-size:.72rem;color:{MUTED}">Intelijen katalog novel</div></div></div>\',',
     'f\'<div style="font-size:.72rem;color:{MUTED}">Novel catalogue intelligence</div></div></div>\','),
    ('choice = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")',
     'choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")'),
    ('''            "<b>4.461</b> judul · NovelUpdates<br>"
            "Rating tertimbang Bayesian<br>"
            "Model dipilih lewat validasi silang</div>",''',
     '''            "<b>4,461</b> titles · NovelUpdates<br>"
            "Bayesian weighted rating<br>"
            "Models selected by cross-validation</div>",'''),
    ('st.error("Layanan data sedang tidak dapat dijangkau. Silakan coba beberapa saat lagi.")',
     'st.error("The data service is unreachable right now. Please try again shortly.")'),
    ('st.warning("Data untuk halaman ini belum tersedia.")',
     'st.warning("The data for this page is not available yet.")'),
]

REG_PAIRS = [
    ('result["diprediksi"] = result["probability"] >= result["threshold"]',
     'result["predicted"] = result["probability"] >= result["threshold"]'),
    ('result.sort_values(["diprediksi", "margin"], ascending=[False, False])',
     'result.sort_values(["predicted", "margin"], ascending=[False, False])'),
]


def apply(path, pairs):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    missing = []
    for old, new in pairs:
        if old not in text:
            missing.append(old.splitlines()[0][:88])
            continue
        text = text.replace(old, new)
    if missing:
        print("GAGAL — %d pola tidak ditemukan di %s:" % (len(missing), path))
        for m in missing:
            print("   !", m)
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print("OK  %s — %d pola diterapkan" % (path, len(pairs)))
    return True


ok = apply(REG, REG_PAIRS) and apply(APP, APP_PAIRS)
sys.exit(0 if ok else 1)
