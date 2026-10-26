# Adil Nöbet

En fazla 8 kişilik ekipler için iki birimde adil nöbet listeleri hazırlayan Python 3.12 ve Streamlit uygulaması. Nöbetleri planlama, elle düzenleme, analiz etme ve dışa aktarma özellikleri sunar.

## Çalıştırma

Depoyu indirip kök dizininde şu komutları çalıştırın (Python 3.12):

```sh
python -m pip install -r requirements.txt
python -m streamlit run nobet.py
```

Tarayıcıda `http://localhost:8501` adresini açın. Kalıcı kayıt için PostgreSQL bağlantısını `DATABASE_URL` ortam değişkeni veya `.streamlit/secrets.toml` içindeki `DATABASE_URL` ile tanımlayın. Bağlantı bilgilerini depoya eklemeyin.

## Testler

```sh
python -m pip install pytest playwright pyflakes
python -m playwright install chromium
python -m pytest -p no:cacheprovider
```

Tarayıcı testleri Playwright ve Chromium kullanır. Gerçek veritabanı gerektiren testler için ayrı bir test PostgreSQL veritabanının bağlantısını `TEST_DATABASE_URL` ile tanımlayın. GitHub Actions üzerindeki **Testler** iş akışı gerekli test ortamını hazırlar.
