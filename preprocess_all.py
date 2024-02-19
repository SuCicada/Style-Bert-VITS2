from webui_train import preprocess_all
import yaml

# 上でつけたフォルダの名前`Data/{model_name}/`
model_name = "lain"
# JP-Extra （日本語特化版）を使うかどうか。日本語の能力が向上する代わりに英語と中国語は使えなくなります。
use_jp_extra = True
# 学習のバッチサイズ。VRAMのはみ出具合に応じて調整してください。
batch_size = 4

# 学習のエポック数（データセットを合計何周するか）。
# 100ぐらいで十分かもしれませんが、もっと多くやると質が上がるのかもしれません。
epochs = 100

# 保存頻度。何ステップごとにモデルを保存するか。分からなければデフォルトのままで。
save_every_steps = 1000

# 音声ファイルの音量を正規化するかどうか。`True`もしくは`False`
normalize = True

# 音声ファイルの開始・終了にある無音区間を削除するかどうか
trim = False

preprocess_all(
    model_name=model_name,
    batch_size=batch_size,
    epochs=epochs,
    save_every_steps=save_every_steps,
    num_processes=2,
    normalize=normalize,
    trim=trim,
    freeze_EN_bert=False,
    freeze_JP_bert=False,
    freeze_ZH_bert=False,
    freeze_style=False,
    use_jp_extra=use_jp_extra,
    val_per_lang=0,
    log_interval=200,
)

model_name = "lain"
with open("default_config.yml", "r", encoding="utf-8") as f:
    yml_data = yaml.safe_load(f)
yml_data["model_name"] = model_name
with open("config.yml", "w", encoding="utf-8") as f:
    yaml.dump(yml_data, f, allow_unicode=True)
