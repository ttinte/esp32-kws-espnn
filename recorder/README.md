# INMP441 recorder

Project ESP-IDF **độc lập** để thu negatives ("other") qua đúng mic INMP441 của
thiết bị, vá lỗi mic-mismatch (dataset thu mic laptop nhưng chạy INMP441).

I2S + phép `>>16` trong [main/recorder_main.cpp](main/recorder_main.cpp) **khớp
chính xác** `../main/kws_engine.cpp`. Nếu sửa I2S ở firmware chính thì sửa luôn ở đây.

## Nạp firmware recorder
```bash
source ~/.esp/esp-idf/export.sh
cd recorder
idf.py set-target esp32s3
idf.py -p <PORT> flash      # KHONG mo monitor (de PC script chiem cong)
```

## Thu trên PC
```bash
cd recorder/tools
pip install pyserial scipy numpy   # neu chua co
python capture_serial.py --port <PORT> --seconds 60 --prefix inmp_sil
```
- Mỗi cảnh chạy 1 lần, `--prefix` khác nhau: im lặng (`inmp_sil`), bật quạt
  (`inmp_fan`), nói chuyện nền (`inmp_talk`)...
- Clip 1s tự lưu vào `kws_voice/kws_dataset/other/`.

## Sau khi thu xong
Nạp lại firmware KWS chính rồi train lại:
```bash
cd ../../kws_voice/scripts
python 05_prepare_wav.py --clean
python 06_build_tf_dataset.py
python 07_train_cnn_kws.py --seed 0
python 09_convert_to_cpp.py
```
