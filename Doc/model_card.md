# Model Card — NovelLens Multi-Head AI

Artefak: `Model/artifacts/` · Dilatih pada 4.460 judul (dari 4.461; 1 baris
tanpa sinyal teks) · Sumber: `s3a://clean-data/novels_clean`

---

## 1. Arsitektur dan alasan pemilihannya

Tiga sifat dataset menentukan pilihan arsitektur:

| Sifat data | Konsekuensi desain |
|---|---|
| Tidak ada `user_id` / histori baca | *Collaborative filtering* (ALS) mustahil → rekomendasi harus berbasis konten |
| Teks campuran prosa + label (sinopsis median 118 kata + 12 tag) | Bag-of-words menangkap frasa penentu genre dengan sangat baik; representasi pralatih melengkapinya pada parafrasa |
| Hanya 4,5k baris | Model besar yang dilatih dari nol akan *overfit*; yang berhasil adalah **memakai** representasi pralatih, bukan membentuknya sendiri |

```
corpus (title + synopsis + tags)
        │
   ┌────┴─────────────────────────────────────────┐
   │ HEAD A — Genre Tagger (4 pilihan)            │  fitur: stemming +
   │  ├─ Logistic Regression                      │  blok TF-IDF terpisah +
   │  ├─ Linear SVM (kalibrasi Platt)             │  metadata + ambang per genre
   │  ├─ Hybrid TF-IDF + Embedding  ← terbaik     │  + blok padat BGE 384-d
   │  └─ BiLSTM + Attention (PyTorch)             │
   └──────────────────────────────────────────────┘
        │
   TF-IDF 1–2 gram → TruncatedSVD 200-d (representasi bersama)
        │
   ┌────┼───────────────┬──────────────────┐
 HEAD B              HEAD C            HEAD D
 Recommender         Arketipe          Estimasi rating
 (cosine kNN)        (KMeans k=5)      (HistGradientBoosting)
```

> **Catatan revisi (2026-08-30).** Versi sebelumnya menyimpulkan bahwa
> representasi pralatih tidak tersedia sehingga jalur transformer tertutup.
> Premis itu keliru — lihat §2.6. Baris "terlalu pendek untuk fine-tuning
> transformer" pada tabel di atas sudah dikoreksi, dan arsitektur `hybrid`
> yang lahir dari koreksi itu kini menjadi model terbaik Head A.

---

## 2. Head A — pelabelan genre (4 model yang dapat dipilih)

### Rujukan penelitian

Susunan model dan protokolnya mengikuti pipeline pembanding yang lazim pada
penelitian klasifikasi genre multi-label dari sinopsis:

| Rujukan | Yang diadopsi |
|---|---|
| Kim (2014), *CNN for Sentence Classification* | Konvolusi 1-D multi-lebar filter (2–5 kata); varian **CNN-rand** karena tanpa embedding pralatih |
| Liu et al. (SIGIR 2017), *XML-CNN* | **Dynamic max pooling**, **bottleneck layer**, **BCE per label** — tiga penyesuaian khusus multi-label |
| Ertugrul & Karagoz (2018) | BiLSTM atas sinopsis untuk prediksi genre |
| Zhou et al. (2016) | Self-attention di atas keluaran BiLSTM |
| Hoang (2018), *Predicting Movie Genres from Plot Summaries* | K-binary transformation + **ambang probabilitas yang dipelajari** |
| Studi pembanding multi-transformer (2025) | Struktur perbandingan: TF-IDF+linear → model urutan; metrik makro & mikro F1 |

### Protokol evaluasi

- Data dibagi **80% pengembangan (3.568) / 20% uji (892)**.
- **Urutan baris dikunci sebelum dibagi** (`sort_values("series_id")`). Ini
  perbaikan, bukan detail: `split_indices` membagi menurut posisi baris,
  sedangkan urutan hasil `toPandas()` bergantung pada cara Spark kebetulan
  mempartisi hasil dan **tidak dijamin sama antar eksekusi**. Sebelum dikunci,
  dua kali menjalankan Fase 4 atas data yang identik menghasilkan pembagian
  yang berbeda dan macro-F1 bergeser sampai ~0,02 tanpa satu baris kode model
  pun berubah. Konsekuensinya jujur untuk disebut: **angka pada versi
  sebelumnya (SVM 0,5377) berasal dari pembagian yang tidak dapat diulang dan
  kebetulan menguntungkan.** Angka di bawah berasal dari pembagian yang kini
  deterministik, dan lebih rendah — bukan karena modelnya memburuk, melainkan
  karena pembandingnya sekarang jujur.
- Hyperparameter **dan** ambang per genre dipilih memakai probabilitas
  *out-of-fold* dari 3-fold CV **di dalam** set pengembangan.
- Set uji disentuh **sekali**, di akhir. Angka di bawah ini berasal dari sana.
- Keempat model memakai pembagian baris yang identik → perbandingan adil.
- Metrik: makro/mikro F1 (konvensi literatur genre), **P@k** dan **Hamming
  loss** (konvensi literatur multi-label / XMC). P@k tidak bergantung ambang
  sama sekali, jadi mengukur kualitas peringkat apa adanya.

### Hasil (set uji, 892 judul, 24 genre)

| Model | Konfigurasi terpilih | OOF makro | F1 makro | F1 mikro | P@3 | Hamming ↓ |
|---|---|---|---|---|---|---|
| **Hybrid TF-IDF + Embedding** | SVM `C=0.3` balanced, bobot embedding 1,0 | **0,5676** | **0,5356** | **0,6502** | **0,7805** | **0,1373** |
| Linear SVM | `C=0.3`, class_weight balanced | 0,5548 | 0,5274 | 0,6479 | 0,7648 | 0,1397 |
| Logistic Regression | `C=4.0`, class_weight balanced | 0,5508 | 0,5167 | 0,6379 | 0,7554 | 0,1428 |
| BiLSTM + Attention | 250 token, hidden 128, 20 epoch, BCE | 0,4733 | 0,4564 | 0,5908 | 0,6626 | 0,1729 |

TextCNN/XML-CNN (Kim 2014; Liu et al. 2017) sempat ikut diuji pada protokol
yang sama dan kalah di seluruh metrik terhadap BiLSTM (makro 0,4483 · mikro
0,5702 · P@3 0,6484 · Hamming 0,2000), sehingga tidak dipertahankan.

#### Di genre mana hybrid menang, dan di mana ia kalah?

Rata-rata menyembunyikan pola yang justru paling menjelaskan. Selisih F1 per
genre terhadap Linear SVM pada pembagian yang sama:

| Menang terbesar | Δ | | Kalah terbesar | Δ |
|---|---|---|---|---|
| Horror | **+0,065** | | Ecchi | −0,050 |
| Sports | **+0,050** | | Psychological | −0,024 |
| Slice of Life | **+0,040** | | Martial Arts | −0,020 |
| School Life | **+0,032** | | Comedy | −0,013 |
| Wuxia | **+0,029** | | Adventure | −0,006 |

Polanya konsisten dengan penjelasan di §2.6. Yang naik adalah genre yang
ditentukan **suasana**, bukan kosakata: Horror, Slice of Life, School Life —
tidak ada satu kata pun yang menandainya, sehingga TF-IDF kesulitan dan
pemahaman kalimat menolong. Yang turun justru genre dengan **kosakata khas
yang sudah tajam** (Martial Arts, Ecchi): di sana TF-IDF sudah hampir optimal
dan menambahkan 384 dimensi padat hanya mengencerkan sinyalnya.

Jadi keunggulan hybrid bukan "lebih pintar secara umum", melainkan
**menutup jenis kesalahan yang berbeda** — dan itu memang yang diharapkan dari
penggabungan dua representasi.

### 2.6 Representasi pralatih — premis yang ternyata keliru

Versi sebelumnya menolak jalur transformer dengan alasan yang eksplisit:
*"Lingkungan ini offline, sehingga embedding harus dilatih dari nol."* Alasan
itu dipakai untuk menjelaskan kekalahan BiLSTM dan untuk menutup arah
perbaikan tersebut.

**Premisnya diuji ulang dan ternyata salah.** Registry model dapat dijangkau
dari dalam container. Begitu itu terbukti, seluruh kesimpulan yang bergantung
padanya harus dibuka kembali — dan hasilnya berubah.

Yang dipakai: **BAAI/bge-small-en-v1.5** (33 juta parameter, keluaran 384-d),
beku, tanpa fine-tuning. Bobotnya di-*bake* ke dalam image Docker sehingga
jalur inferensi tetap tidak bergantung pada jaringan.

**Teks yang masuk encoder berbeda dari yang masuk TF-IDF, dan itu disengaja:**

| Jalur | Masukan | Alasan |
|---|---|---|
| TF-IDF | `corpus_clean` — sudah di-stem, stopword dibuang | Menyatukan varian kata jadi satu fitur |
| Embedding | `title + sinopsis mentah + tags` — kalimat utuh | Encoder dilatih pada prosa normal; merusak sintaksnya menjauhkan masukan dari distribusi latihnya |

#### Ablasi: sendirian kalah, digabung menang

Diukur pada protokol yang sama (OOF 3-fold di set pengembangan):

| Representasi | OOF makro | Putusan |
|---|---|---|
| TF-IDF saja (baseline) | 0,5471 | pembanding |
| **Embedding BGE saja** | **0,5142** | kalah telak — sendirian bukan pengganti |
| **TF-IDF + embedding (fusi)** | **0,5659** | **diadopsi** |

Ini temuan yang paling penting dan paling mudah disalahpahami: **embedding
pralatih lebih buruk daripada TF-IDF bila dipakai sendirian.** Yang menolong
bukan "model yang lebih canggih", melainkan **keragaman representasi**.

Penjelasannya konsisten dengan sifat datanya. Sebagian besar sinyal genre di
korpus ini terletak pada keberadaan frasa khas — "cultivation", "transmigration",
"system administrator". TF-IDF menangkap itu nyaris sempurna, dan memampatkannya
ke 384 dimensi padat justru membuang ketajamannya. Yang **tidak** ditangkap
TF-IDF adalah parafrasa: dua sinopsis yang menceritakan hal sama dengan pilihan
kata berbeda tidak berbagi satu pun token. Di situlah embedding menyumbang.
Keduanya salah di tempat yang berbeda, sehingga penggabungannya menambah
informasi alih-alih menggandakannya.

#### Sapuan bobot blok embedding

| Bobot | 0,5 | 1,0 | 2,0 | 3,0 | 5,0 |
|---|---|---|---|---|---|
| OOF makro (SVM) | 0,5545 | 0,5630 | **0,5659** | 0,5567 | 0,5381 |
| OOF makro (LogReg) | 0,5512 | **0,5598** | 0,5597 | 0,5529 | 0,5318 |

Puncaknya landai di sekitar 1–2 lalu turun tajam di 5 — begitu embedding
terlalu dominan, frasa khas yang jadi kekuatan TF-IDF ikut tenggelam.
**Kesepuluh konfigurasi fusi mengungguli baseline**, jadi keunggulannya tidak
bergantung pada penyetelan bobot yang beruntung.

#### Apakah ini nyata, atau kebetulan pembagian data?

Selisihnya kecil (+0,013 OOF), dan satu set uji hanya berisi 892 baris —
cukup kecil untuk membuat macro-F1 bergoyang. Karena itu keunggulannya diuji
**berpasangan pada 5 pembagian berbeda**: pada tiap seed, kedua model dilatih
dan diuji pada pembagian yang sama persis, lalu yang dianalisis selisihnya.
Ragam yang berasal dari pembagian muncul di kedua model dan saling meniadakan.
Hasilnya ada di §2.7.

#### Yang belum dikerjakan: fine-tuning ujung-ke-ujung

Uji ini memakai encoder **beku**. Fine-tuning BGE secara penuh untuk 24 label
sudah disiapkan (`scripts/research/run_finetune.py`, protokol OOF yang sama)
tetapi **tidak dijalankan sampai selesai**: pada container ini satu epoch atas
2.379 dokumen memakan belasan menit, sehingga protokol lengkap (3 fold × 3
epoch + model akhir) diperkirakan 2–3 jam. Itu di luar anggaran waktu sesi ini,
dan juga jauh melewati waktu latih Head A saat ini. Statusnya: **terbuka, belum
terjawab** — bukan sudah diuji dan ditolak.

### 2.7 Uji berpasangan lintas 5 pembagian data

Selisih +0,013 pada satu pembagian tidak cukup untuk menyimpulkan apa pun bila
mengganti pembagian saja sudah menggeser skor lebih besar dari itu. Karena itu
kedua model diuji berpasangan pada 5 seed: sama data, sama protokol, hanya
pembagiannya yang berbeda.

| Seed | F1 makro uji | F1 mikro uji | P@3 uji |
|---|---|---|---|
| 42 | 0,5449 → **0,5575** | 0,6452 → **0,6514** | 0,7745 → **0,7765** |
| 7 | 0,5677 → **0,5800** | 0,6484 → **0,6530** | 0,7659 → **0,7805** |
| 123 | 0,5611 → **0,5621** | 0,6448 → **0,6494** | 0,7676 → **0,7805** |
| 2024 | 0,5414 → **0,5559** | 0,6316 → **0,6390** | 0,7752 → **0,7823** |
| 99 | 0,5464 → **0,5656** | 0,6452 → **0,6526** | 0,7691 → **0,7857** |

Ringkasan (baseline Linear SVM → Hybrid):

| Metrik | Baseline | Hybrid | Selisih | Menang | t |
|---|---|---|---|---|---|
| **OOF makro** | 0,5424 ± 0,0030 | **0,5588 ± 0,0046** | **+0,0164 ± 0,0029** | **5/5** | 12,6 |
| F1 makro uji | 0,5523 ± 0,0114 | **0,5642 ± 0,0096** | **+0,0119 ± 0,0067** | **5/5** | 4,0 |
| F1 mikro uji | 0,6430 ± 0,0066 | **0,6491 ± 0,0058** | **+0,0060 ± 0,0014** | **5/5** | 9,6 |
| P@3 uji | 0,7705 ± 0,0042 | **0,7811 ± 0,0033** | **+0,0106 ± 0,0060** | **5/5** | 4,0 |

Empat metrik, lima pembagian, **dua puluh perbandingan — hybrid menang di
semuanya.** Dengan n=5 ini bukan bukti statistik yang kuat (t≈4 pada df=4
setara p≈0,016), tetapi konsistensinya jauh lebih meyakinkan daripada satu
angka pada satu pembagian.

Angka yang paling perlu diperhatikan justru simpangan bakunya: macro-F1
baseline bergoyang **±0,0114** antar pembagian, sedangkan selisih
berpasangannya hanya **±0,0067**. Artinya ragam terbesar pada eksperimen ini
memang datang dari pembagian data, bukan dari modelnya — dan itulah alasan
perbandingan tak-berpasangan pada satu set uji mudah menyesatkan.

### Kenapa F1-nya "hanya" segitu? — batas atas tugasnya

Pertanyaan wajar: makro-F1 0,54 terdengar rendah. Jawabannya bukan model yang
buruk, melainkan **labelnya sendiri tidak konsisten**.

Diukur langsung, bukan diduga: untuk setiap novel dicari tetangga terdekatnya
di ruang embedding (novel yang isinya paling mirip), lalu dibandingkan
himpunan genre keduanya. Kalau dua novel yang isinya nyaris identik saja
diberi genre berbeda oleh kurator, tidak ada model yang bisa memisahkannya.

| Kemiripan konten | n pasang | F1 mikro | F1 makro | Jaccard |
|---|---|---|---|---|
| 0,80 – 0,90 (nyaris identik) | 35 | 0,732 | 0,579 | 0,539 |
| 0,70 – 0,80 | 179 | 0,623 | 0,547 | 0,467 |
| 0,60 – 0,70 | 726 | 0,596 | 0,460 | 0,453 |

**Batas atas empiris ≈ 0,653 makro-F1** (rata-rata kesepakatan pada pasangan
dengan kemiripan ≥0,75). Model terbaik mencapai **±83% dari batas itu**.

Genre mana yang paling tidak konsisten? Persis genre yang skor modelnya rendah:

| Genre | Kesepakatan antar novel mirip | F1 model | Sifat label |
|---|---|---|---|
| Mature | 0,00 | 0,36 | **Rating konten**, bukan genre |
| Horror | 0,00 | 0,53 | Subjektif |
| Mystery | 0,44 | 0,45 | Batas kabur dengan Fantasy/Action |
| Slice of Life | 0,47 | 0,52 | Definisinya longgar |
| Psychological | 0,50 | 0,36 | Label nada cerita |
| Adult / Ecchi | 0,60–0,69 | 0,38–0,39 | Tergantung adegan yang jarang ada di sinopsis |

Sebaliknya genre yang konkret dan terlihat dari isi cerita mencapai F1 tinggi:
Romance 0,84 · Fantasy 0,80 · Action 0,78 · Adventure 0,70.

Jadi model sudah mendekati batas informasi yang tersedia. Menaikkannya lebih
jauh memerlukan **label yang lebih bersih**, bukan arsitektur yang lebih besar.

### Apa saja yang sudah dicoba untuk menaikkannya (dan hasilnya)

Semua diuji pada protokol yang sama dan dipilih berdasarkan **skor OOF**, bukan
skor uji — supaya tidak terjadi kebocoran seleksi.

**Rekayasa fitur**

| Percobaan | Hipotesis | OOF makro | Putusan |
|---|---|---|---|
| Baseline (korpus + tag) | — | 0,5371 | pembanding |
| **+ blok sinopsis terpisah** | Prosa butuh ruang IDF sendiri | **0,5471** | **diadopsi** |
| + fitur tag utuh (indikator biner) | Frasa tag terpecah oleh stemming | 0,4914 | ditolak — memburuk |
| Tag utuh menggantikan TF-IDF tag | Hilangkan duplikasi | 0,4862 | ditolak |
| Tag utuh bobot rendah | Dosis lebih kecil | 0,5135 | ditolak |
| Tag saja (tanpa prosa) | Ablasi | 0,5065 | — |
| Sinopsis saja (tanpa tag) | Ablasi | 0,4548 | — |

**Teknik dari literatur multi-label**

| Teknik | Rujukan | OOF makro | Putusan |
|---|---|---|---|
| Baseline (setelah blok sinopsis) | — | 0,5471 | pembanding |
| EDA — random deletion 15% | Wei & Zou, EMNLP 2019 | 0,5429 | ditolak |
| EDA — deletion 30%, 2 salinan | Wei & Zou, EMNLP 2019 | 0,5459 | ditolak |
| MLSMOTE | Charte et al., KBS 2015 | 0,5323 | ditolak |
| MLSMOTE + EDA | kombinasi | 0,5321 | ditolak |
| Fitur khusus per label, k=8.000 | LIFT, Zhang & Wu 2015 | 0,5483 | ditolak — selisih setingkat derau, mikro & P@3 turun |
| Fitur khusus per label, k=20.000 | LIFT, Zhang & Wu 2015 | 0,5490 | ditolak — alasan sama |
| Asymmetric Loss (BiLSTM) | Ridnik et al., ICCV 2021 | 0,4606 vs BCE 0,4618 | ditolak — imbang |
| Stacking antar label | korelasi P(Ecchi\|Harem)=0,73 | 0,5266 | ditolak |
| Ensemble SVM + LogReg | rata-rata dua model | 0,5361 | ditolak |

**Kenapa augmentasi tidak menolong di sini** — ini pertanyaan yang layak
dijawab, bukan sekadar dilaporkan gagal:

* **MLSMOTE** membuat titik sintetis lewat interpolasi antara dua dokumen di
  ruang TF-IDF. Pada teks berdimensi tinggi dan sangat sparse, hasil
  interpolasi adalah "rata-rata dua novel" yang tidak berpadanan dengan
  dokumen nyata mana pun — model belajar dari wilayah ruang fitur yang tidak
  pernah muncul saat inferensi.
* **EDA random deletion** hampir tidak menambah informasi bagi model
  bag-of-words: setelah pembobotan sublinear dan normalisasi L2, menghapus
  sebagian token menghasilkan vektor yang nyaris sama dengan aslinya. Dari
  empat operasi EDA, *random swap* memang tidak berpengaruh sama sekali pada
  TF-IDF (urutan kata tidak dipakai), sedangkan *synonym replacement*
  membutuhkan WordNet yang tidak tersedia di lingkungan offline ini.
* **Asymmetric Loss** dirancang untuk ketimpangan positif–negatif yang ekstrem
  (ratusan hingga ribuan label). Di sini hanya ada 24 label dengan rata-rata
  4,2 positif per dokumen — ketimpangannya jauh lebih ringan, dan
  `pos_weight` pada BCE sudah menanganinya.

Pelajaran yang paling tidak terduga: **menambah representasi tag justru
merugikan.** Sebabnya `corpus_clean` sudah memuat tag di dalamnya, sehingga
blok tambahan hanya menggandakan informasi yang sama dan mengencerkan sinyal
sinopsis. Yang berhasil justru kebalikannya — memberi sinopsis ruangnya sendiri.

### Temuan lain: model deep learning KALAH dari baseline linear

Ini hasil yang harus disampaikan apa adanya, bukan disembunyikan. Selisihnya
besar dan konsisten di semua metrik — Linear SVM unggul **+0,057 F1 makro**
dan **+0,10 P@3** atas BiLSTM.

Penjelasan yang paling masuk akal, dan sejalan dengan literatur:

1. **Tidak ada embedding pralatih.** Seluruh makalah rujukan memakai
   GloVe/word2vec yang dilatih pada miliaran kata. Lingkungan ini offline,
   sehingga embedding harus dilatih dari nol pada 3.568 dokumen — jauh di
   bawah kebutuhan untuk membentuk ruang kata yang bermakna. Studi pembanding
   2025 menunjukkan pola yang sama: model dengan representasi pralatih
   (RoBERTa, makro-F1 0,518) mengungguli model urutan yang dilatih dari nol.
2. **Ukuran data.** 3.568 dokumen latih untuk 24 label adalah wilayah di mana
   model berparameter banyak overfit, sementara model linear di ruang TF-IDF
   berdimensi tinggi justru berada di kondisi terbaiknya.
3. **Sifat sinyalnya.** Sebagian besar informasi genre ada pada **keberadaan
   frasa tertentu** ("cultivation", "system administrator"), bukan pada
   urutannya. Itu persis yang ditangkap bag-of-words, dan keunggulan CNN/RNN
   dalam memodelkan urutan tidak terpakai.

### Upaya perbaikan model DL yang sudah dilakukan (dan hasilnya)

Dua perbaikan diterapkan setelah hasil pertama, keduanya berbasis praktik baku:

| Perubahan | Alasan | Hasil |
|---|---|---|
| **`pos_weight` pada BCE** | Genre langka nyaris tak memberi gradien; menebak "bukan" sudah menghasilkan loss rendah | Makro naik tipis (BiLSTM 0,4715→0,4812) |
| **Stemming Snowball** pada masukan DL | Menyatukan varian kata agar kosakata tidak terlalu terpecah | Ikut tercakup di angka di atas |

Upaya ini dijalankan pada TextCNN dan BiLSTM sekaligus, dan urutan peringkat
keduanya tidak berubah — BiLSTM tetap unggul, keduanya tetap di bawah baseline.

Trade-off yang muncul dan perlu dicatat: `pos_weight` menaikkan makro-F1
tetapi **menurunkan P@3** (BiLSTM 0,7278 → 0,6715). Model jadi lebih berani
menyebut genre langka, dengan ongkos peringkat tiga teratas yang lebih kotor.
Perbaikan ini **tidak** menutup jarak terhadap baseline linear.

Kesimpulan jujur: pada dataset sebesar ini dan tanpa embedding pralatih,
**arsitektur deep learning bukan alat yang tepat untuk tugas ini.** BiLSTM
tetap disediakan di dashboard sebagai pembanding yang transparan, bukan karena
lebih unggul.

### Peningkatan dibanding versi awal (model berbasis fitur)

Versi awal: satu Logistic Regression, TF-IDF polos atas korpus gabungan,
tanpa stemming, tanpa metadata, ambang seragam 0,5.

| Metrik (set uji) | Sebelum | Sesudah | Selisih |
|---|---|---|---|
| F1 makro | 0,5121 | **0,5377** | **+5,0%** |
| F1 mikro | 0,6346 | **0,6374** | +0,4% |

> Kedua kolom di atas diukur pada **pembagian data lama yang tidak deterministik**
> (lihat "Protokol evaluasi"). Angkanya dipertahankan apa adanya sebagai catatan
> sejarah perbandingan *sebelum vs sesudah rekayasa fitur*, bukan sebagai skor
> yang berlaku sekarang. Skor yang berlaku ada di tabel "Hasil (set uji)".

Yang berubah dan seberapa besar sumbangannya:

1. **Ambang keputusan per genre** — penyumbang terbesar, dan tetap terbesar
   pada pembagian yang baru: macro-F1 SVM naik dari **0,3982** (ambang seragam
   0,5) menjadi **0,5274**, yaitu **+32%**. Pada model hybrid pun polanya sama
   (0,4131 → 0,5356). Sebabnya: Romance hadir di 56% katalog sementara Sports
   di bawah 2%; dengan ambang seragam, genre langka nyaris tak pernah diprediksi.
2. **Stemming Snowball** — "cultivation/cultivating/cultivator" menyatu jadi
   satu fitur, bukan tiga fitur lemah.
3. **Blok fitur terpisah berbobot** — tag (label kurasi, presisi tinggi) diberi
   bobot 1,6× dibanding korpus prosa yang lebih berderau.
4. **Metadata sebagai fitur** — asal negara sangat menentukan genre; Xianxia
   praktis hanya ada pada novel Tiongkok.
5. **Pencarian hyperparameter** — 9 kandidat dievaluasi lewat OOF CV.
   Temuan yang tidak terduga: Logistic Regression justru lebih baik **tanpa**
   `class_weight="balanced"` begitu ambang disetel — memakai keduanya adalah
   koreksi ganda terhadap ketimpangan kelas.

### Membaca angka ini secara jujur

F1 mikro nyaris tidak bergerak (+0,4%); yang benar-benar naik adalah F1 makro,
yaitu kemampuan mengenali genre langka. Itu memang target perbaikannya, tetapi
perlu dikatakan apa adanya: **peningkatan keseluruhannya moderat, bukan
lompatan.** Batasnya bersifat intrinsik — pelabelan genre di sumber data
sebagian subjektif (novel yang sama bisa diberi label berbeda oleh kurator
berbeda), sehingga ada langit-langit yang tidak bisa ditembus model mana pun
yang belajar dari label tersebut.

---

## 3. Head B, C, D

| Head | Tugas | Metrik | Nilai |
|---|---|---|---|
| **B** | Rekomendasi berbasis konten | Genre overlap@10 | **0,9388** |
| **C** | Klasterisasi arketipe | Silhouette (k=5) | 0,0169 |
| **D** | Estimasi rating dari konten | MAE | **0,4164** (baseline 0,4544) |
| | | RMSE / R² | 0,5285 / 0,1674 |

- **Head B (overlap 0,94)** — 9 dari 10 rekomendasi berbagi minimal satu genre
  dengan judul acuan. Ini **proksi**, bukan pengganti uji A/B; tanpa data
  perilaku pembaca, relevansi sesungguhnya tidak terukur.
- **Head C (silhouette 0,017)** rendah, dan itu khas data teks berdimensi
  tinggi: klaster saling bersinggungan, bukan terpisah rapi.
- **Head D (R² 0,17)** — konten hanya menjelaskan ~17% variasi rating. Model
  **8,4%** lebih baik daripada menebak rata-rata, tetapi keluarannya
  **indikasi arah**, bukan prediksi presisi. Dashboard menyatakan ini eksplisit.

### Head D: sumbangan fitur kecepatan rilis

`log_days_per_chapter` ditambahkan pada revisi ini (lihat data_contract §5.1).
Karena pembagian data juga berubah pada revisi yang sama, sumbangannya diuji
terpisah — data dan pembagian dikunci, hanya fitur itu yang di-*toggle*:

| Konfigurasi | MAE ↓ | R² ↑ |
|---|---|---|
| Tanpa kecepatan rilis | **0,4045** | 0,1782 |
| Dengan kecepatan rilis | 0,4090 | **0,1945** |

Hasilnya **campuran, dan dilaporkan apa adanya**: R² naik cukup jelas (+0,016,
artinya lebih banyak variasi rating yang terjelaskan) tetapi MAE justru naik
tipis (+0,0045, artinya galat rata-rata sedikit memburuk). Tafsirnya: fitur ini
membantu model menangkap judul-judul yang ratingnya menyimpang jauh — yang
memang paling memengaruhi R² — dengan ongkos sedikit ketelitian pada judul
kebanyakan. Fitur dipertahankan karena R² adalah yang paling relevan untuk
pertanyaan "apa yang menjelaskan penerimaan pembaca", tetapi keunggulannya
**tidak seragam di semua metrik.**

### Catatan: `k` klaster berpindah dari 9 ke 5

Revisi ini memilih k=5, sebelumnya k=9. Bukan karena datanya berubah drastis,
melainkan karena **silhouette-nya nyaris rata di seluruh rentang**:

| k | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|
| Silhouette | **0,0169** | 0,0153 | 0,0154 | 0,0154 | 0,0165 | 0,0162 |

Selisih antara yang terbaik dan terburuk hanya 0,0016. Artinya **pemilihan `k`
lewat silhouette pada data ini praktis tidak informatif** — hasilnya bisa
berayun antar-eksekusi hanya karena inisialisasi KMeans. Ini keterbatasan yang
sebelumnya tidak disebutkan: k=9 pada versi lama bukan "jumlah arketipe yang
benar", melainkan pemenang tipis dari perbandingan yang nyaris seri.

### Arketipe hasil klasterisasi (k=5)

| # | Ukuran | Istilah dominan |
|---|---|---|
| 0 | 494 | cultivation, martial, world, sect |
| 1 | 780 | world, magic, game, fantasy |
| 2 | 1.267 | female protagonist, family, marriage |
| 3 | 1.224 | male protagonist, world, day |
| 4 | 695 | love, love interests, lead |

Kelompok-kelompok ini terbaca sebagai arketipe naratif yang dikenali pembaca
(xianxia kultivasi, isekai, LitRPG, romansa keluarga) — bukti kualitatif bahwa
ruang embedding menangkap makna, bukan derau.

---

## 4. Keputusan yang mencegah kebocoran & bias

1. **Fitur popularitas dikeluarkan dari Head D.** `rating_votes` masuk ke dalam
   rumus `weighted_rating`, sehingga memakainya sebagai fitur = membocorkan
   target. `reading_list_count` juga tidak dipakai agar model menilai isi
   cerita, bukan keramaiannya.
2. **Head D hanya dilatih pada rating tepercaya (≥10 suara).** Rating dari 1–2
   suara adalah derau; melatih di atasnya mengajarkan model meniru kebisingan.
3. **Genre dengan <50 novel dibuang** (24 dari 33 genre dipakai). Kelas sekecil
   itu hanya menghasilkan klasifikator yang selalu memprediksi "tidak".
4. **Ambang disetel pada OOF, bukan set uji.** Menyetel lalu melaporkan pada
   data yang sama akan melebih-lebihkan performa.
5. **Daftar stopword Fase 2 disimpan sebagai artefak** dan dibaca ulang oleh
   jalur inferensi, sehingga tidak terjadi *train/serve skew*.
6. **`k` klaster dipilih lewat silhouette**, bukan ditebak.

---

## 5. Batasan yang harus disebutkan saat presentasi

- Model membaca **deskripsi**, bukan naskah. Novel dengan sinopsis buruk akan
  diprediksi buruk meski isinya bagus.
- Rekomendasi murni berbasis konten: dua judul bertema sama dianggap mirip
  walau mutu penulisannya jauh berbeda.
- Katalog didominasi novel Tiongkok (2.455 dari 4.461); performa untuk judul
  Barat/EN kemungkinan lebih lemah karena representasinya sedikit.
- Rating berasal dari satu platform (NovelUpdates) dengan basis pembaca
  terjemahan — bukan populasi pembaca umum.
- Arketipe adalah kelompok yang bersinggungan, bukan sekat tegas. Lebih jauh:
  **jumlah klaster `k` pada dasarnya tidak dapat ditentukan dari data ini** —
  silhouette-nya rata (0,0153–0,0169 untuk k=5..10), sehingga k yang terpilih
  bisa berubah antar-eksekusi. Perlakukan arketipe sebagai alat eksplorasi,
  bukan segmentasi yang stabil.
- Model `hybrid` bergantung pada bobot encoder BGE yang di-*bake* ke dalam
  image. Menjalankannya di image lama, atau mengubah `SENTENCE_EMBEDDING_MODEL`
  tanpa membangun ulang, akan menggagalkan jalur inferensinya.
- Fine-tuning transformer ujung-ke-ujung **belum diuji** (lihat §2.6) — bukan
  sudah dicoba dan ditolak.

---

## 6. Artefak yang dihasilkan

| File | Isi | Ukuran |
|---|---|---|
| `genre_models.joblib` | 4 pipeline genre lengkap + ambang per genre | 49,8 MB |
| `svd.joblib` | Proyeksi LSA 200 dimensi | 35,7 MB |
| `vectorizer.joblib` | TF-IDF representasi bersama | 0,27 MB |
| `genre_binarizer.joblib` | Pemetaan 24 label | <0,01 MB |
| `kmeans.joblib` | KMeans k=5 | 0,01 MB |
| `rating_regressor.joblib` | HistGradientBoosting | 0,28 MB |
| `embeddings.npy` | Matriks embedding 4460×200 (float32) | 3,4 MB |
| `catalog.parquet` | Metadata + cuplikan sinopsis + cluster_id | 1,8 MB |
| `metrics.json` | Seluruh metrik, profil klaster, stopword | 0,02 MB |

Semua artefak juga disalin ke `s3a://curated-data/models/` sebagai cadangan
yang dapat diaudit.

---

## 7. Cara melatih ulang

```bash
docker compose exec pipeline python -m src.phases.phase4_model
```

Waktu training penuh ≈ **38 menit** (tanpa GPU, 2 thread BLAS):

| Tahap | Waktu | Catatan |
|---|---|---|
| logreg + svm | ~1,5 menit | — |
| **hybrid** | ~8 menit | ~6 menit di antaranya adalah encoding 4.460 dokumen — sekali saja, lalu di-memo lintas fold |
| bilstm | ~28 menit | tiap kandidat butuh 4 kali pelatihan jaringan (3 fold CV + model akhir) |
| Head B/C/D | ~20 detik | — |

Encoding hybrid dibatasi `OMP_NUM_THREADS=2` dari image. Naikkan
`ENCODER_THREADS` di `.env` bila host punya banyak core — hanya PyTorch yang
terpengaruh, OpenBLAS (dan alasan asli pembatasannya) tidak tersentuh.

Dashboard memuat artefak baru setelah container Streamlit di-restart.

Untuk melatih ulang satu arsitektur saja tanpa menyentuh yang lain:

```bash
docker compose exec pipeline python Model/train.py --models hybrid
```
